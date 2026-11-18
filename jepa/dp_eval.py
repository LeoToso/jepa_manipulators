#!/usr/bin/env python
"""Offline evaluation: Diffusion Policy alone vs. Diffusion Policy steered by each world model.

    python -m jepa.dp_eval --dp runs/dp --cache /mnt/t7shield/cache_jepa_128 --out eval/dp_steer \\
        --model "MSP+SIG=runs/msp_sigreg" --model "MSP+GCB+SIG=runs/msp_gcb_sigreg" \\
        --model "MSP+SIG+SF=runs/msp_sigreg_sf_lam10"

At every --stride-th frame t of each held-out episode (val + test by default):
  1. the policy proposes N action chunks (16 steps = 1.6 s) from the real observation at t;
  2. each world model encodes frame t, rolls every chunk out (8 latent steps) and keeps the chunk
     whose predicted end state is best for a goal image:
        latent distance |zhat_end - z_goal|^2 (every model), and for successor models also the
        goal value V_g(zhat_end) = <S(u, u_g), u_g> / d;
  3. the chosen chunk is compared with the recorded actions over the same 1.6 s.
Goal images (--goals): final  = last frame of the episode (deployment setting: a goal photo)
                       subgoalK = the frame K latent steps after t (K * frame_skip frames)
                       oracle = the frame at the end of the chunk (upper bound, not a fair result)
Reference selections without a world model:
  policy      the expected error of a single sample (mean over the N candidates)
  consensus   the candidate closest to all others (gain from choosing among N, no world model)
  best        the candidate closest to the recorded actions (the best any selection can do)
Errors of a chunk: end-point displacement error (cm), end-point rotation error (deg, composed
rotations), gripper command error (% of steps on the wrong side of the open/close midpoint).
Writes results.npz, summary.md/json and fig_steer_<goal>.png/pdf, fig_steer_summary.png/pdf.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from jepa.data import Cache, quat_conj, quat_mul, task_family  # noqa: E402
from jepa.evaluate import Evaluator  # noqa: E402
from jepa.policy import PolicyData, load_policy  # noqa: E402

MODEL_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False, 'figure.facecolor': 'white',
    'savefig.facecolor': 'white', 'lines.linewidth': 1.8, 'pdf.fonttype': 42,
})


def rotvec_to_quat(r):
    ang = np.linalg.norm(r, axis=-1, keepdims=True)
    axis = r / np.maximum(ang, 1e-12)
    return np.concatenate([np.cos(ang / 2), axis * np.sin(ang / 2)], axis=-1)


def compose(chunk):
    """(..., T, 7) per-step actions -> total displacement (..., 3) and rotation quaternion (..., 4)
    (base-frame deltas: q_{t+1} = dq_t * q_t)."""
    dpos = chunk[..., :3].sum(-2)
    dq = rotvec_to_quat(chunk[..., 3:6].astype(np.float64))
    q = np.zeros(chunk.shape[:-2] + (4,))
    q[..., 0] = 1.0
    for k in range(chunk.shape[-2]):
        q = quat_mul(dq[..., k, :], q)
    return dpos, q


def chunk_errors(cand, gt, grip_mid):
    """cand (..., T, 7), gt broadcastable -> pos [cm], rot [deg], gripper [% steps]."""
    pc, qc = compose(cand)
    pg, qg = compose(gt)
    pos = 100 * np.linalg.norm(pc - pg, axis=-1)
    w = np.abs(quat_mul(qc, quat_conj(qg))[..., 0]).clip(max=1.0)
    rot = np.degrees(2 * np.arccos(w))
    grip = 100 * ((cand[..., 6] > grip_mid) != (gt[..., 6] > grip_mid)).mean(-1)
    return pos, rot, grip


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dp', required=True, help='policy.pt or the folder that holds it')
    p.add_argument('--cache', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model', action='append', default=[], help='LABEL=RUN_DIR (uses RUN_DIR/model_final.pt)')
    p.add_argument('--n-samples', type=int, default=32)
    p.add_argument('--ddim-steps', type=int, default=16)
    p.add_argument('--stride', type=int, default=8, help='evaluate every stride-th frame')
    p.add_argument('--goals', default='final,subgoal15,oracle',
                   help='final | oracle | subgoalK (own future) | demoK (another demonstration) | nnK '
                        '(retrieved from the nearest training states) | gatedK (final goal when within K '
                        'latent steps, else policy alone) | hybridK (final when near, else nnK)')
    p.add_argument('--gate-fallback', default='consensus', choices=['consensus', 'policy'],
                   help='gatedK: what is executed while the final goal is out of reach')
    p.add_argument('--pair', action='append', default=[],
                   help='"RESULTS_DIR:GOAL:TITLE" (give two): only draw fig_drift_pair side by side into --out')
    p.add_argument('--nn-k', type=int, default=5, help='demonstrations retrieved for nnK / hybridK goals')
    p.add_argument('--gate-pct', type=float, default=75.0,
                   help='gatedK / hybridK: percentile of K-step latent distances that counts as "within reach"')
    p.add_argument('--splits', default='val,test')
    p.add_argument('--batch-states', type=int, default=64)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--replot', action='store_true', help='only redraw the figures from results.npz')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.pair:
        drift_pair([spec.split(':', 2) for spec in a.pair], a.out)
        return
    if a.replot:
        R = dict(np.load(os.path.join(a.out, 'results.npz'), allow_pickle=True))
        meta = json.load(open(os.path.join(a.out, 'meta.json')))
        report(R, meta, a.out)
        return
    dev = torch.device(a.device if (a.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)

    # ---- policy and held-out states ---------------------------------------------------------
    pol, ck = load_policy(a.dp if a.dp.endswith('.pt') else os.path.join(a.dp, 'policy.pt'), dev)
    c = Cache(a.cache)
    if list(c.tasks) != list(ck['tasks']):
        raise SystemExit(f'task list of the cache {c.tasks} differs from the policy {ck["tasks"]}')
    H = pol.horizon
    pd = PolicyData(c, H, ck['obs_skip'], ck['stats']['proprio_keys'], stats=ck['stats'],
                    action_mode=ck.get('action_mode', 'delta_next'))
    eps = [e for s in a.splits.split(',') for e in c.split_episodes(s)]
    T = np.concatenate([np.arange(c.starts[e], c.starts[e] + c.lengths[e] - 1, a.stride) for e in eps])
    n = len(T)
    ep = c.episode_of_frame[T]
    progress = (T - c.starts[ep]) / np.maximum(c.lengths[ep] - 1, 1)
    task = c.episode_task[ep]
    family = np.array([task_family(c.tasks[t]) for t in task])
    gt = pd.actions[pd.action_rows(T)]                                            # (n, H, 7) raw
    grip_mid = 0.5 * (pd.a_min[6] + pd.a_max[6])
    N = a.n_samples
    print(f'{len(eps)} held-out episodes, {n} states, {N} candidates each; policy horizon {H}', flush=True)

    # ---- 1. candidates from the policy (shared by every selection method) -----------------------
    cand_n = np.zeros((n, N, H, 7), np.float32)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    for i in range(0, n, a.batch_states):
        t = T[i:i + a.batch_states]
        img, prop, tk, _ = pd.batch(t, dev)
        with torch.no_grad():
            cond = pol.encode_obs(img, prop, tk).repeat_interleave(N, 0)
            x = pol.sample(cond, a.ddim_steps, generator=g)
        cand_n[i:i + len(t)] = x.reshape(len(t), N, H, 7).cpu().numpy()
        print(f'  sampled {min(i + a.batch_states, n)}/{n}', end='\r', flush=True)
    print()
    cand = pd.unnormalize(cand_n)                                                 # raw per-step
    pos, rot, grip = chunk_errors(cand, gt[:, None], grip_mid)                    # (n, N)

    R = {'t': T, 'episode': ep, 'task': task, 'family': family, 'progress': progress}
    methods = []

    R.update({'cand|pos': pos.astype(np.float32), 'cand|rot': rot.astype(np.float32),
              'cand|grip': grip.astype(np.float32)})

    ex = min(a.stride, H)  # steps executed per chunk before the next decision (for the drift analysis)
    R.update({'gt': gt.astype(np.float32), 'cand_exec': cand[:, :, :ex].astype(np.float32)})

    def record(name, idx=None, goal='-'):
        if idx is None:  # expected error of one sample; its drift uses candidate 0 (one random sample)
            vals = (pos.mean(1), rot.mean(1), grip.mean(1))
            R[f'{name}|{goal}|idx'] = np.zeros(n, np.int64)
        elif isinstance(idx, str) and idx == 'min':  # lowest error among the N, metric by metric
            vals = (pos.min(1), rot.min(1), grip.min(1))
            R[f'{name}|{goal}|idx'] = pos.argmin(1)
            R[f'{name}|{goal}|idx_rot'] = rot.argmin(1)
        else:
            r = np.arange(n)
            vals = (pos[r, idx], rot[r, idx], grip[r, idx])
            R[f'{name}|{goal}|idx'] = np.asarray(idx)
        for k, v in zip(('pos', 'rot', 'grip'), vals):
            R[f'{name}|{goal}|{k}'] = v
        methods.append((name, goal))

    record('policy')
    flat = cand_n.reshape(n, N, -1)
    cons = np.zeros(n, np.int64)
    for i in range(0, n, 256):  # candidate with the smallest summed distance to the others
        f = flat[i:i + 256]
        cons[i:i + 256] = ((f[:, :, None] - f[:, None]) ** 2).mean(-1).sum(-1).argmin(1)
    record('consensus', cons)
    record('best', 'min')  # a true lower bound for every metric (not one shared candidate)

    # ---- 2. world-model selection -------------------------------------------------------------
    goals = a.goals.split(',')
    ep_end = pd.ep_end[T]

    def goal_frames(name):
        if name == 'final':
            return ep_end
        if name == 'oracle':
            return np.minimum(T + H, ep_end)
        if name.startswith('subgoal') or name.startswith('demo'):
            return None  # needs the model's frame_skip
        if name.startswith(('subgoal', 'demo', 'nn', 'gated', 'hybrid')):
            return None
        raise SystemExit(f'unknown goal {name!r}')

    # reference demonstrations for the demoK goals: one training episode of the same task per
    # held-out episode (fixed seed); the subgoal is the frame at the same episode progress, K latent
    # steps further on, in that other demonstration
    tr_by_task = {ti: [e for e in c.split_episodes('train') if c.episode_task[e] == ti] for ti in range(len(c.tasks))}
    ref_rng = np.random.RandomState(a.seed + 1)
    ref_of = {int(e): int(ref_rng.choice(tr_by_task[int(c.episode_task[e])])) for e in eps}
    ref = np.array([ref_of[int(e)] for e in ep])

    def demo_frames(K, fs):
        rs, rl = c.starts[ref], c.lengths[ref]
        k = np.round(progress * (rl - 1)).astype(np.int64) + K * fs
        return rs + np.clip(k, 0, rl - 1)

    frame_skip = None
    for spec in a.model:
        label, run = spec.split('=', 1)
        ckpt = run if run.endswith('.pt') else os.path.join(run, 'model_final.pt')
        ev = Evaluator(ckpt, a.cache, a.device, batch_size=2048)
        if ev.dc['action_mode'] != 'delta_next' or H % ev.fs:
            raise SystemExit(f'{label}: needs action_mode delta_next and frame_skip dividing {H}')
        st = ev.ckpt['action_stats']
        wm_act = ((cand - np.array(st['mean'], np.float32)) / np.array(st['std'], np.float32))
        wm_act = wm_act.reshape(n * N, H // ev.fs, 7 * ev.fs).astype(np.float32)
        z0 = ev.encode(T)
        zhat = ev.rollout(np.repeat(z0, N, 0), wm_act)[:, -1].reshape(n, N, -1)  # predicted end latents
        has_sf = getattr(ev.model, 'successor', None) is not None
        frame_skip = ev.fs
        print(f'{label}: rolled out {n * N} chunks ({H // ev.fs} latent steps)'
              + (' [successor value available]' if has_sf else ''), flush=True)
        def sf_values(zg):
            """(n, N) goal values of the predicted end latents; zg (n, d) or (n, k, d) (mean over k)."""
            zg3 = zg[:, None] if zg.ndim == 2 else zg
            out_v = np.zeros((n, N), np.float32)
            with torch.no_grad():
                for i in range(0, n, 64):
                    zz = torch.from_numpy(zhat[i:i + 64]).to(ev.device)                     # (b, N, d)
                    for j in range(zg3.shape[1]):
                        gg = torch.from_numpy(zg3[i:i + 64, j]).to(ev.device)[:, None].expand_as(zz)
                        out_v[i:i + 64] += ev.model.goal_value(
                            zz.reshape(-1, zz.shape[-1]), gg.reshape(-1, gg.shape[-1])).reshape(zz.shape[:2]).float().cpu().numpy()
            return out_v / zg3.shape[1]

        def dist_cost(zg):
            zg3 = zg[:, None] if zg.ndim == 2 else zg
            return np.stack([((zhat - zg3[:, None, j]) ** 2).mean(-1) for j in range(zg3.shape[1])]).mean(0)

        cache_goal = {}

        def retrieval_goals(K):
            """(n, k, d): for each state, the k training frames of the same task nearest to z_t in this
            model's latent (at most one per demonstration), each advanced by K latent steps in its own
            demonstration. Uses only training data, so it is deployable."""
            key = f'nn{K}'
            if key in cache_goal:
                return cache_goal[key]
            if 'train' not in cache_goal:
                tf = ev.episode_frames(c.split_episodes('train'))
                cache_goal['train'] = (tf, ev.encode(tf))
            tf, ztr = cache_goal['train']
            tf_ep = c.episode_of_frame[tf]
            tf_task = c.episode_task[tf_ep]
            out_g = np.zeros((n, a.nn_k, z0.shape[1]), np.float32)
            for ti in np.unique(task):
                si = np.flatnonzero(task == ti)
                fi = np.flatnonzero(tf_task == ti)
                d = (z0[si] ** 2).sum(1)[:, None] + (ztr[fi] ** 2).sum(1)[None] - 2 * z0[si] @ ztr[fi].T
                eps_t = np.unique(tf_ep[fi])
                best_f = np.zeros((len(si), len(eps_t)), np.int64)
                best_d = np.zeros((len(si), len(eps_t)))
                for j, e in enumerate(eps_t):
                    cols = np.flatnonzero(tf_ep[fi] == e)
                    am = d[:, cols].argmin(1)
                    best_f[:, j] = fi[cols[am]]
                    best_d[:, j] = d[np.arange(len(si)), cols[am]]
                pick = np.argsort(best_d, 1)[:, :a.nn_k]
                nb = np.take_along_axis(best_f, pick, 1)                                   # rows of tf
                gframe = np.minimum(tf[nb] + K * ev.fs, pd.ep_end[tf[nb]])
                lookup = {int(f): i for i, f in enumerate(tf)}
                out_g[si] = ztr[np.vectorize(lookup.get)(gframe)]
            cache_goal[key] = out_g
            return out_g

        def gate_open(K):
            """Steer towards the final frame only when it is within reach: latent distance from z_t to
            the final frame below the --gate-pct percentile of distances between training frames K
            latent steps apart in the same task (calibrated per task, training data only)."""
            key = f'gate{K}'
            if key in cache_goal:
                return cache_goal[key]
            if 'train' not in cache_goal:
                tf = ev.episode_frames(c.split_episodes('train'))
                cache_goal['train'] = (tf, ev.encode(tf))
            tf, ztr = cache_goal['train']
            pos_of = {int(f): i for i, f in enumerate(tf)}
            nxt = np.minimum(tf + K * ev.fs, pd.ep_end[tf])
            valid = nxt - tf == K * ev.fs
            dk = ((ztr - ztr[np.vectorize(pos_of.get)(nxt)]) ** 2).mean(1)
            tf_task = c.episode_task[c.episode_of_frame[tf]]
            zf = final_latent()
            d0 = ((z0 - zf) ** 2).mean(1)
            thr = np.array([np.percentile(dk[valid & (tf_task == ti)], a.gate_pct) for ti in task])
            cache_goal[key] = d0 <= thr
            return cache_goal[key]

        def final_latent():
            if 'final' not in cache_goal:
                cache_goal['final'] = ev.encode(ep_end)
            return cache_goal['final']

        for gname in goals:
            if gname.startswith(('nn', 'gated', 'hybrid')):
                K = int(''.join(ch for ch in gname if ch.isdigit()))
                cost_f = dist_cost(final_latent())
                if gname.startswith('nn'):
                    zg_nn = retrieval_goals(K)
                    record(label, dist_cost(zg_nn).argmin(1), gname)
                    if has_sf:
                        record(f'{label} (value)', sf_values(zg_nn).argmax(1), gname)
                    continue
                open_ = gate_open(K)
                R[f'{label}|{gname}|gate'] = open_
                if gname.startswith('gated'):  # far from the goal: consensus (or policy alone)
                    alt_d = alt_v = cons if a.gate_fallback == 'consensus' else np.zeros(n, np.int64)
                else:                           # hybrid: retrieval subgoal while the goal is far
                    zg_nn = retrieval_goals(K)
                    alt_d = dist_cost(zg_nn).argmin(1)
                    alt_v = sf_values(zg_nn).argmax(1) if has_sf else None
                record(label, np.where(open_, cost_f.argmin(1), alt_d), gname)
                if has_sf:
                    record(f'{label} (value)', np.where(open_, sf_values(final_latent()).argmax(1), alt_v), gname)
                print(f'  {label} {gname}: final goal used at {100 * open_.mean():.0f}% of states', flush=True)
                continue
            gf = goal_frames(gname)
            if gf is None and gname.startswith('subgoal'):
                gf = np.minimum(T + int(gname[len('subgoal'):]) * ev.fs, ep_end)
            elif gf is None:
                gf = demo_frames(int(gname[len('demo'):]), ev.fs)
            zg = ev.encode(gf)
            record(label, dist_cost(zg).argmin(1), gname)
            if has_sf:
                record(f'{label} (value)', sf_values(zg).argmax(1), gname)
        del ev
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

    meta = {'methods': methods, 'goals': goals, 'models': [s.split('=', 1)[0] for s in a.model],
            'n_states': int(n), 'n_episodes': len(eps), 'n_samples': N, 'horizon': H, 'stride': a.stride,
            'splits': a.splits, 'frame_skip': frame_skip, 'executed_steps': int(ex), 'control_hz': 10.0}
    np.savez(os.path.join(a.out, 'results.npz'), **R)
    with open(os.path.join(a.out, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    report(R, meta, a.out)


# --------------------------------------------------------------------------- #
# Summary and figures
# --------------------------------------------------------------------------- #

def style(name, models):
    """(colour, linestyle, label) of a selection method."""
    if name == 'policy':
        return INK, '-', 'policy alone (1 sample)'
    if name == 'consensus':
        return INK2, '--', 'consensus of N (no world model)'
    if name == 'best':
        return INK2, ':', 'lower bound (best of N, per metric)'
    base = name.replace(' (value)', '')
    col = MODEL_COLORS[models.index(base) % len(MODEL_COLORS)] if base in models else INK2
    if name.endswith('(value)'):
        return col, '-', f'policy + {base}, goal value'
    return col, ('--' if any(f'{base} (value)' == m for m, _ in METHODS) else '-'), f'policy + {base}' + (
        ', latent distance' if any(f'{base} (value)' == m for m, _ in METHODS) else '')


METHODS = []


def per_episode_band(x, prog, ep, bins):
    """Mean over states in each progress bin, and the standard error over episodes."""
    b = np.clip(np.digitize(prog, bins) - 1, 0, len(bins) - 2)
    mean, se = np.full(len(bins) - 1, np.nan), np.full(len(bins) - 1, np.nan)
    for k in range(len(bins) - 1):
        m = b == k
        if not m.any():
            continue
        per_ep = np.array([x[m & (ep == e)].mean() for e in np.unique(ep[m])])
        mean[k] = per_ep.mean()
        se[k] = per_ep.std(ddof=1) / np.sqrt(len(per_ep)) if len(per_ep) > 1 else 0.0
    return mean, se


def report(R, meta, out):
    global METHODS
    METHODS = [tuple(m) for m in meta['methods']]
    models = meta['models']
    fams = sorted(set(R['family'].tolist()))
    metrics = (('pos', 'end-point position error [cm]'), ('rot', 'end-point rotation error [deg]'),
               ('grip', 'gripper command error [% steps]'))
    # summary table
    rows, summ = [], {}
    for goal in meta['goals']:
        for name, g in METHODS:
            if g not in ('-', goal):
                continue
            key = f'{name}|{g}'
            entry = {}
            for f in fams + ['all']:
                m = np.ones(len(R['family']), bool) if f == 'all' else R['family'] == f
                entry[f] = {k: float(R[f'{key}|{k}'][m].mean()) for k, _ in metrics}
            summ.setdefault(goal, {})[name] = entry
            rows.append([goal, name] + [f'{entry[f]["pos"]:.2f} / {entry[f]["rot"]:.2f} / {entry[f]["grip"]:.1f}'
                                        for f in fams + ['all']])
    with open(os.path.join(out, 'summary.json'), 'w') as fh:
        json.dump(summ, fh, indent=1)
    head = ['goal', 'method'] + [f'{f}: cm / deg / grip%' for f in fams + ['all']]
    md = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(r) + ' |' for r in rows]
    open(os.path.join(out, 'summary.md'), 'w').write(
        f'{meta["n_episodes"]} held-out episodes ({meta["splits"]}), {meta["n_states"]} states, '
        f'{meta["n_samples"]} candidates, chunk {meta["horizon"]} steps. Lower is better.\n\n' + '\n'.join(md) + '\n')
    print('\n'.join(md))

    # error along the trajectory, one figure per goal
    bins = np.linspace(0, 1, 11)
    mid = 0.5 * (bins[1:] + bins[:-1])
    for goal in meta['goals']:
        fig, axes = plt.subplots(len(fams), 3, figsize=(13, 3.3 * len(fams)), squeeze=False)
        handles = {}
        for r, f in enumerate(fams):
            m = R['family'] == f
            for c_, (k, ylab) in enumerate(metrics):
                ax = axes[r, c_]
                for name, g in METHODS:
                    if g not in ('-', goal):
                        continue
                    col, ls, lab = style(name, models)
                    mean, se = per_episode_band(R[f'{name}|{g}|{k}'][m], R['progress'][m], R['episode'][m], bins)
                    h, = ax.plot(mid, mean, color=col, ls=ls, marker='o', ms=3,
                                 lw=2.2 if name not in ('consensus', 'best') else 1.5)
                    ax.fill_between(mid, mean - se, mean + se, color=col, alpha=0.12, linewidth=0)
                    handles[lab] = h
                ax.set_xlim(0, 1)
                ax.set_ylim(bottom=0)
                ax.set_title(f'{f.replace("_", "-")}', fontsize=9.5)
                ax.set_ylabel(ylab, fontsize=8.5)
                if r == len(fams) - 1:
                    ax.set_xlabel('episode progress')
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.legend(list(handles.values()), list(handles.keys()), loc='lower center', ncol=3, fontsize=8.5)
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(out, f'fig_steer_{goal}.{ext}'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    # summary bars: mean error over all states, per goal setting
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    sel = [(name, g) for name, g in METHODS if g != '-']
    names = list(dict.fromkeys(nm for nm, _ in sel))
    refs = [nm for nm, g in METHODS if g == '-']
    width = 0.8 / max(len(names), 1)
    for ax, (k, ylab) in zip(axes, metrics):
        for i, goal in enumerate(meta['goals']):
            for j, nm in enumerate(names):
                col, ls, _ = style(nm, models)
                v = R[f'{nm}|{goal}|{k}'].mean()
                ax.bar(i - 0.4 + width * (j + 0.5), v, width * 0.92, color=col, alpha=1.0 if ls == '-' else 0.55,
                       hatch='//' if ls != '-' else None, edgecolor='white', linewidth=0.5)
        for nm in refs:
            col, ls, lab = style(nm, models)
            ax.axhline(R[f'{nm}|-|{k}'].mean(), color=col, ls=ls, lw=1.4)
        ax.set_xticks(range(len(meta['goals'])), [f'goal: {g}' for g in meta['goals']])
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.grid(axis='x', visible=False)
    hs = [plt.Rectangle((0, 0), 1, 1, color=style(nm, models)[0], alpha=1.0 if style(nm, models)[1] == '-' else 0.55)
          for nm in names] + [plt.Line2D([], [], color=style(nm, models)[0], ls=style(nm, models)[1]) for nm in refs]
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(hs, [style(nm, models)[2] for nm in names] + [style(nm, models)[2] for nm in refs],
               loc='lower center', ncol=4, fontsize=8.5)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'fig_steer_summary.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    main_figures(R, meta, out)
    horizon_figure(R, meta, out)
    paired_stats(R, meta, out)
    if 'gt' in R:
        practical_table(R, meta, out)
        drift_figures(R, meta, out)
    print(f'[done] {out}')


KEY_PANELS = (('pick_and_place', 'pos', 'pick-and-place: position error [cm]'),
              ('pick_and_place', 'grip', 'pick-and-place: gripper command error [% steps]'),
              ('rotate', 'rot', 'rotate: rotation error [deg]'))


def _key_panels(R):
    fams = set(R['family'].tolist())
    return [k for k in KEY_PANELS if k[0] in fams]


def _save(fig, out, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main_figures(R, meta, out):
    """fig_steer_main_<goal>: the three informative panels only (pick-and-place position and
    gripper, rotate rotation) along the episode."""
    models, bins = meta['models'], np.linspace(0, 1, 11)
    mid = 0.5 * (bins[1:] + bins[:-1])
    panels = _key_panels(R)
    for goal in meta['goals']:
        fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6), squeeze=False)
        handles = {}
        for ax, (fam, k, ylab) in zip(axes[0], panels):
            m = R['family'] == fam
            for name, g in METHODS:
                if g not in ('-', goal):
                    continue
                col, ls, lab = style(name, models)
                mean, se = per_episode_band(R[f'{name}|{g}|{k}'][m], R['progress'][m], R['episode'][m], bins)
                h, = ax.plot(mid, mean, color=col, ls=ls, marker='o', ms=3,
                             lw=2.2 if name not in ('consensus', 'best') else 1.5)
                ax.fill_between(mid, mean - se, mean + se, color=col, alpha=0.12, linewidth=0)
                handles[lab] = h
            ax.set_xlim(0, 1)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('episode progress')
            ax.set_ylabel(ylab, fontsize=8.5)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.legend(list(handles.values()), list(handles.keys()), loc='lower center', ncol=3, fontsize=8.5)
        _save(fig, out, f'fig_steer_main_{goal}')


def horizon_figure(R, meta, out):
    """fig_steer_horizon: mean error vs how far ahead the subgoal is (subgoal goals only)."""
    subs = sorted([g for g in meta['goals'] if g.startswith('subgoal')], key=lambda g: int(g[len('subgoal'):]))
    if len(subs) < 2:
        return
    fs = meta.get('frame_skip') or 2
    secs = np.array([int(g[len('subgoal'):]) * fs / 10.0 for g in subs])
    models, panels = meta['models'], _key_panels(R)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6), squeeze=False)
    handles = {}
    for ax, (fam, k, ylab) in zip(axes[0], panels):
        m = R['family'] == fam
        for name, g in METHODS:
            col, ls, lab = style(name, models)
            if g == '-':
                h = ax.axhline(R[f'{name}|-|{k}'][m].mean(), color=col, ls=ls, lw=1.4)
                handles[lab] = h
        for name in dict.fromkeys(nm for nm, g in METHODS if g in subs):
            col, ls, lab = style(name, models)
            vals = [R[f'{name}|{g}|{k}'][m].mean() if f'{name}|{g}|{k}' in R else np.nan for g in subs]
            h, = ax.plot(secs, vals, color=col, ls=ls, marker='o', ms=4, lw=2.0)
            handles[lab] = h
        ax.set_xticks(secs, [f'{x:g}' for x in secs])
        ax.set_ylim(bottom=0)
        ax.set_xlabel('subgoal distance ahead [s]')
        ax.set_ylabel(ylab, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(list(handles.values()), list(handles.keys()), loc='lower center', ncol=3, fontsize=8.5)
    _save(fig, out, 'fig_steer_horizon')


def paired_stats(R, meta, out, n_boot=2000, seed=0):
    """Paired comparisons on the same states and candidates: mean difference A - B with a 95%
    bootstrap interval over episodes (episodes resampled with replacement) and a two-sided p."""
    rng = np.random.RandomState(seed)
    panels = [(f, k, lab.split(':')[0] + ' ' + {'pos': 'cm', 'rot': 'deg', 'grip': '% grip'}[k])
              for f, k, lab in _key_panels(R)]
    rows, js = [], {}
    for goal in meta['goals']:
        wm = [nm for nm, g in METHODS if g == goal][::-1]  # successor models first: A = SF
        pairs = [(a_, b_) for i, a_ in enumerate(wm) for b_ in wm[i + 1:]]
        pairs += [(a_, ref) for a_ in wm for ref in ('consensus', 'policy')]
        for fam, k, plab in panels:
            m = R['family'] == fam
            eps = R['episode'][m]
            ue, inv = np.unique(eps, return_inverse=True)
            cnt = np.bincount(inv)
            for a_, b_ in pairs:
                ka = f'{a_}|{goal}|{k}'
                kb = f'{b_}|{"-" if b_ in ("consensus", "policy", "best") else goal}|{k}'
                d = R[ka][m] - R[kb][m]
                sums = np.bincount(inv, weights=d)
                draw = rng.randint(0, len(ue), (n_boot, len(ue)))
                boot = sums[draw].sum(1) / cnt[draw].sum(1)
                diff = d.mean()
                lo, hi = np.percentile(boot, [2.5, 97.5])
                pval = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
                rows.append([goal, f'{a_} vs {b_}', plab, f'{R[ka][m].mean():.2f}', f'{R[kb][m].mean():.2f}',
                             f'{diff:+.2f} [{lo:+.2f}, {hi:+.2f}]', f'{pval:.3f}' if pval >= 0.001 else '<0.001'])
                js.setdefault(goal, []).append({'a': a_, 'b': b_, 'panel': plab, 'mean_a': float(R[ka][m].mean()),
                                                'mean_b': float(R[kb][m].mean()), 'diff': float(diff),
                                                'ci95': [float(lo), float(hi)], 'p': float(pval)})
    head = ['goal', 'comparison (A vs B)', 'metric', 'A', 'B', 'A - B [95% CI]', 'p']
    md = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(r) + ' |' for r in rows]
    open(os.path.join(out, 'stats.md'), 'w').write(
        'Paired differences on the same states and candidates (negative = A has lower error). 95% CI and '
        f'p from {n_boot} bootstrap resamples of episodes.\n\n' + '\n'.join(md) + '\n')
    with open(os.path.join(out, 'stats.json'), 'w') as fh:
        json.dump(js, fh, indent=1)


def practical_table(R, meta, out):
    """practical.md: errors relative to how much the hand moves in one chunk, and errors at the
    critical moments (pick-and-place chunks in which the gripper opens or closes; the last 20% of
    rotate episodes)."""
    gt = R['gt']
    dpos, q = compose(gt)
    motion_cm = 100 * np.linalg.norm(dpos, axis=-1)
    motion_deg = np.degrees(2 * np.arccos(np.abs(q[..., 0]).clip(max=1.0)))
    grip_mid = 0.5 * (gt[..., 6].min() + gt[..., 6].max())
    switch = ((gt[..., 6] > grip_mid)[:, 1:] != (gt[..., 6] > grip_mid)[:, :-1]).any(1)
    fam = R['family']
    pp, ro = fam == 'pick_and_place', fam == 'rotate'
    crit_pp = pp & switch
    crit_ro = ro & (R['progress'] >= 0.8)
    lines = [f'Hand motion per chunk ({meta["horizon"]} steps = {meta["horizon"] / meta["control_hz"]:g} s): '
             f'pick-and-place {motion_cm[pp].mean():.2f} cm (median {np.median(motion_cm[pp]):.2f}), '
             f'rotate {motion_deg[ro].mean():.2f} deg (median {np.median(motion_deg[ro]):.2f}).',
             f'Critical states: {int(crit_pp.sum())} pick-and-place chunks with a gripper open/close, '
             f'{int(crit_ro.sum())} rotate states in the last 20% of the episode.', '']
    head = ['goal', 'method', 'p&p position, cm (% of motion)', 'rotate rotation, deg (% of motion)',
            'grasp/release chunks: position cm', 'grasp/release chunks: gripper %', 'end of rotate: rotation deg']
    rows = []
    for goal in meta['goals']:
        for name, g in METHODS:
            if g not in ('-', goal):
                continue
            k = f'{name}|{g}|'
            p_, r_, g_ = R[k + 'pos'], R[k + 'rot'], R[k + 'grip']
            rows.append([goal, name,
                         f'{p_[pp].mean():.2f} ({100 * p_[pp].mean() / max(motion_cm[pp].mean(), 1e-9):.0f}%)',
                         f'{r_[ro].mean():.2f} ({100 * r_[ro].mean() / max(motion_deg[ro].mean(), 1e-9):.0f}%)',
                         f'{p_[crit_pp].mean():.2f}' if crit_pp.any() else '-',
                         f'{g_[crit_pp].mean():.1f}' if crit_pp.any() else '-',
                         f'{r_[crit_ro].mean():.2f}' if crit_ro.any() else '-'])
    md = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)] + ['| ' + ' | '.join(r) + ' |' for r in rows]
    open(os.path.join(out, 'practical.md'), 'w').write('\n'.join(lines + md) + '\n')


def _integrate(steps):
    """(S, 7) per-step actions -> cumulative position (S+1, 3) and orientation quaternion (S+1, 4)."""
    pos = np.concatenate([np.zeros((1, 3)), np.cumsum(steps[:, :3], 0)])
    dq = rotvec_to_quat(steps[:, 3:6].astype(np.float64))
    q = np.zeros((len(steps) + 1, 4))
    q[0, 0] = 1.0
    for k in range(len(steps)):
        q[k + 1] = quat_mul(dq[k], q[k])
    return pos, q


def drift_curve(R, meta, name, g, fam, k):
    """Accumulated open-loop drift of one selection method: (time [s], mean, s.e. over episodes)."""
    ex, hz = meta['executed_steps'], meta['control_hz']
    gt_exec, cand, ep = R['gt'][:, :ex], R['cand_exec'], R['episode']
    idx_key = f'{name}|{g}|idx_rot' if (k == 'rot' and f'{name}|{g}|idx_rot' in R) else f'{name}|{g}|idx'
    idx = R[idx_key]
    curves = []
    for e in np.unique(ep[R['family'] == fam]):
        st = np.flatnonzero(ep == e)                       # this episode's states, in time order
        pc, qc = _integrate(cand[st, idx[st]].reshape(-1, 7))
        pg, qg = _integrate(gt_exec[st].reshape(-1, 7))
        if k == 'pos':
            err = 100 * np.linalg.norm(pc - pg, axis=-1)
        else:
            err = np.degrees(2 * np.arccos(np.abs(quat_mul(qc, quat_conj(qg))[:, 0]).clip(max=1.0)))
        curves.append(err[::ex])                            # at each decision point
    L = int(np.percentile([len(cv) for cv in curves], 50))   # times reached by >= half the episodes
    M = np.full((len(curves), L), np.nan)
    for i, cv in enumerate(curves):
        M[i, :min(L, len(cv))] = cv[:L]
    mean = np.nanmean(M, 0)
    se = np.nanstd(M, 0) / np.sqrt(np.maximum((~np.isnan(M)).sum(0), 1))
    return np.arange(L) * ex / hz, mean, se


DRIFT_PANELS = (('pick_and_place', 'pos', 'pick-and-place: accumulated position drift [cm]'),
                ('rotate', 'rot', 'rotate: accumulated rotation drift [deg]'))


def _drift_axes(ax, R, meta, goal, fam, k, handles):
    for name, g in METHODS:
        if g not in ('-', goal):
            continue
        tsec, mean, se = drift_curve(R, meta, name, g, fam, k)
        col, ls, lab = style(name, meta['models'])
        h, = ax.plot(tsec, mean, color=col, ls=ls, lw=2.2 if name not in ('consensus', 'best') else 1.5)
        ax.fill_between(tsec, mean - se, mean + se, color=col, alpha=0.12, linewidth=0)
        handles[lab] = h
        yield name, tsec, mean
    ax.set_xlim(0, None)
    ax.set_ylim(bottom=0)


def drift_figures(R, meta, out):
    """fig_steer_drift_<goal>: execute the first `executed_steps` actions of each chosen chunk,
    one after the other over the whole episode, WITHOUT feedback (each decision still sees the
    recorded observation), and integrate them. The gap to the recorded hand pose shows how the
    per-chunk errors accumulate: random errors partly cancel, systematic ones add up. On the robot,
    feedback from the camera would correct part of this drift, so this is a pessimistic view."""
    res = {}
    for goal in meta['goals']:
        panels = [p_ for p_ in DRIFT_PANELS if (R['family'] == p_[0]).any()]
        fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 3.8), squeeze=False)
        handles = {}
        for ax, (fam, k, ylab) in zip(axes[0], panels):
            for name, tsec, mean in _drift_axes(ax, R, meta, goal, fam, k, handles):
                res.setdefault(goal, {}).setdefault(fam, {})[name] = {
                    'time_s': tsec.tolist(), 'mean': mean.tolist(), 'final': float(mean[-1])}
            ax.set_xlabel('time in episode [s] (open loop, no feedback)')
            ax.set_ylabel(ylab, fontsize=8.5)
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        fig.legend(list(handles.values()), list(handles.keys()), loc='lower center', ncol=3, fontsize=8.5)
        _save(fig, out, f'fig_steer_drift_{goal}')
    with open(os.path.join(out, 'drift.json'), 'w') as fh:
        json.dump(res, fh, indent=1)


def drift_pair(specs, out):
    """fig_drift_pair: the drift figure of two (results dir, goal) settings side by side; rows are
    the task families, columns the settings, the y-range shared along each row."""
    global METHODS
    loaded = []
    for d, goal, *title in specs:
        R = dict(np.load(os.path.join(d, 'results.npz'), allow_pickle=True))
        meta = json.load(open(os.path.join(d, 'meta.json')))
        if goal not in meta['goals']:
            raise SystemExit(f'{d}: goal {goal!r} not in {meta["goals"]}')
        loaded.append((R, meta, goal, title[0] if title else f'{os.path.basename(os.path.normpath(d))}: {goal}'))
    panels = [p_ for p_ in DRIFT_PANELS if all((R['family'] == p_[0]).any() for R, *_ in loaded)]
    fig, axes = plt.subplots(len(panels), len(loaded), figsize=(5.6 * len(loaded), 3.6 * len(panels)),
                             squeeze=False, sharey='row')
    handles = {}
    for c_, (R, meta, goal, title) in enumerate(loaded):
        METHODS = [tuple(m) for m in meta['methods']]
        for r, (fam, k, ylab) in enumerate(panels):
            ax = axes[r, c_]
            list(_drift_axes(ax, R, meta, goal, fam, k, handles))
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c_ == 0:
                ax.set_ylabel(ylab, fontsize=8.5)
            if r == len(panels) - 1:
                ax.set_xlabel('time in episode [s] (open loop, no feedback)')
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(list(handles.values()), list(handles.keys()), loc='lower center', ncol=3, fontsize=8.5)
    _save(fig, out, 'fig_drift_pair')
    print(f'[done] {os.path.join(out, "fig_drift_pair.png")}')

if __name__ == '__main__':
    main()
