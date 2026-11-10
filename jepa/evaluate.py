#!/usr/bin/env python
"""Latent-space analysis of one trained world model.

    python -m jepa.evaluate --ckpt runs/msp_sigreg/model_final.pt --cache cache_128 --out eval/msp_sigreg

Everything is computed on held-out test episodes; probes are fitted on training
episodes. Writes <out>/metrics.json and figures in <out>/figs/:

 A  probes.png            what the encoded latent z_t contains: linear / MLP R^2 of physical
                          quantities, task / family / object accuracy (linear + kNN)
 B  family_probes.png     position vs. rotation information, evaluated inside each task family
    predicted_latents.png the same probes applied to k-step predictions zhat_{t+k}, and the
                          relative prediction error vs. k (what survives the predictor)
 C  spectrum.png          covariance eigenvalues, effective rank (overall and per task)
    gaussianity.png       random 1-D projections vs N(0,1) (LeJEPA Fig. 2/5 style)
    embedding_*.png       PCA / t-SNE coloured by task, progress, gripper, EE height
    task_subspaces.png    overlap of per-task principal subspaces (shared vs task-specific)
    temporal.png          straightening and transition-subspace r95 per task
 D  goal_distance.png     latent distance to the episode's final frame vs. progress (planning
                          cost monotonicity) and a steps-to-goal probe
    analogies.png         z(B_0) + [z(A_T) - z(A_0)] -> nearest neighbour (GCB-style analogies)
 E  planning.png          offline CEM towards a recorded goal frame vs. the recorded actions
 F  patch_pca.png         first 3 PCs of the encoder's patch tokens as RGB (LeJEPA Fig. 14)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from jepa.data import (Cache, proprio_array, quat_conj, quat_mul, quat_to_rotvec, task_family,
                       task_object, to_wxyz)
from jepa.models import load_checkpoint
from jepa.probes import (Standardizer, effective_rank, knn_classify, mlp_r2, r2_guarded,
                         ridge_classify, ridge_fit, ridge_r2, r2_score)

QUANTITIES = ['ee_pos', 'ee_rot', 'gripper', 'joints', 'ee_lin_vel', 'ee_ang_vel', 'progress']


# --------------------------------------------------------------------------- #
# Encoding and labels
# --------------------------------------------------------------------------- #

class Evaluator:
    def __init__(self, ckpt_path, cache_dir, device, batch_size=256):
        self.device = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
        self.model, self.ckpt = load_checkpoint(ckpt_path, self.device)
        cfg = self.ckpt['config']
        self.dc = cfg['data']
        self.fs, self.H = int(self.dc['frame_skip']), int(self.dc['horizon'])
        self.cache = Cache(cache_dir, self.dc.get('quat_order', 'wxyz'))
        if list(self.cache.tasks) != list(self.ckpt['tasks']):
            print('[warn] task list of the cache differs from training:', self.cache.tasks)
        self.bs = batch_size
        stats = self.ckpt['action_stats']
        self.actions = ((self.cache.actions(self.dc['action_mode']) - np.array(stats['mean'], np.float32))
                        / np.array(stats['std'], np.float32)).astype(np.float32)
        ps = self.ckpt.get('proprio_stats')
        self.proprio = None
        if ps is not None:  # normalized exactly as in training
            self.proprio = ((proprio_array(self.cache, ps['keys']) - np.array(ps['mean'], np.float32))
                            / np.array(ps['std'], np.float32)).astype(np.float32)
        c = self.cache
        self.fam_names, self.obj_names = c.families, c.objects
        self.task_of_frame = c.episode_task[c.episode_of_frame]
        self.family_of_task = np.array([c.families.index(task_family(t)) for t in c.tasks])
        self.object_of_task = np.array([c.objects.index(task_object(t)) for t in c.tasks])
        self._labels()

    def _labels(self):
        c, low = self.cache, self.cache.low
        train_frames = self.episode_frames(c.split_episodes('train'))
        q = to_wxyz(low['x_ee'][:, 3:7].astype(np.float64), c.quat_order)
        ref = q[train_frames]
        ref = np.where((ref @ ref[0])[:, None] < 0, -ref, ref).mean(0)
        ref /= np.linalg.norm(ref)
        rot = quat_to_rotvec(quat_mul(q, quat_conj(np.broadcast_to(ref, q.shape))))
        self.labels = {
            'ee_pos': low['x_ee'][:, :3], 'ee_rot': rot.astype(np.float32),
            'gripper': low['gripper'][:, :1], 'joints': low['q'], 'ee_lin_vel': low['v_ee'][:, :3],
            'ee_ang_vel': low['v_ee'][:, 3:6],
            'progress': (c.step_in_episode / np.maximum(c.lengths[c.episode_of_frame] - 1, 1))[:, None],
        }

    def episode_frames(self, episodes, stride=1):
        c = self.cache
        return np.concatenate([np.arange(c.starts[e], c.starts[e] + c.lengths[e], stride) for e in episodes])

    def prev_frames(self, frames):
        c = self.cache
        return np.maximum(frames - self.fs, c.starts[c.episode_of_frame[frames]])

    @torch.no_grad()
    def encode(self, frames, return_tokens=False):
        out, toks = [], []
        imgs = self.cache.images
        for i in range(0, len(frames), self.bs):
            f = frames[i:i + self.bs]
            o = torch.from_numpy(np.ascontiguousarray(imgs[f])).permute(0, 3, 1, 2).to(self.device)
            p = torch.from_numpy(np.ascontiguousarray(imgs[self.prev_frames(f)])).permute(0, 3, 1, 2).to(self.device)
            prop = None if self.proprio is None else torch.from_numpy(self.proprio[f]).to(self.device)
            r = self.model.encode(o, p, prop, return_tokens=return_tokens)
            if return_tokens:
                out.append(r[0].float().cpu())
                toks.append(r[1].float().cpu())
            else:
                out.append(r.float().cpu())
        z = torch.cat(out).numpy()
        return (z, torch.cat(toks).numpy()) if return_tokens else z

    def action_chunks(self, starts, k):
        """(n, k, 7*fs) normalized action chunks starting at the given frames."""
        rows = starts[:, None] + np.arange(k * self.fs)[None]
        return self.actions[rows].reshape(len(starts), k, -1)

    @torch.no_grad()
    def rollout(self, z0, chunks):
        out = []
        for i in range(0, len(z0), self.bs):
            out.append(self.model.rollout(torch.from_numpy(z0[i:i + self.bs]).to(self.device),
                                          torch.from_numpy(chunks[i:i + self.bs]).to(self.device)).float().cpu())
        return torch.cat(out).numpy()


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #

def straightening(z):
    dz = np.diff(z, axis=0)
    n = np.linalg.norm(dz, axis=1)
    ok = (n[:-1] > 1e-8) & (n[1:] > 1e-8)
    cos = np.sum(dz[:-1] * dz[1:], axis=1)[ok] / (n[:-1] * n[1:])[ok]
    return float(cos.mean()) if len(cos) else float('nan')


def transition_r95(z, energy=0.95):
    s = np.linalg.svd(np.diff(z, axis=0), compute_uv=False) ** 2
    return int(np.searchsorted(np.cumsum(s) / max(s.sum(), 1e-12), energy) + 1)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def epps_pulley(x, slices=256, seed=0):
    """SIGReg statistic of a fixed set of directions (lower = closer to N(0, I))."""
    rng = np.random.RandomState(seed)
    a = rng.randn(x.shape[1], slices)
    a /= np.linalg.norm(a, axis=0)
    t = np.linspace(0, 3, 17)
    w = np.full(17, 2 * 3 / 16)
    w[[0, -1]] = 3 / 16
    phi = np.exp(-t ** 2 / 2)
    xt = (x @ a)[:, :, None] * t
    err = (np.cos(xt).mean(0) - phi) ** 2 + np.sin(xt).mean(0) ** 2
    return float((err @ (w * phi)).mean() * len(x))


def subspace_overlap(za, zb, k):
    """Mean squared cosine of principal angles between top-k PCA subspaces (1 = identical)."""
    ua = np.linalg.svd(za - za.mean(0), full_matrices=False)[2][:k].T
    ub = np.linalg.svd(zb - zb.mean(0), full_matrices=False)[2][:k].T
    return float(np.sum((ua.T @ ub) ** 2) / k)


def cem_plan(model, z0, zg, horizon, action_dim, device, pop=300, elites=30, iters=10, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    z0 = torch.from_numpy(z0).to(device)[None]
    zg = torch.from_numpy(zg).to(device)[None]
    mu = torch.zeros(horizon, action_dim)
    sd = torch.ones(horizon, action_dim)
    for _ in range(iters):
        cand = mu + sd * torch.randn(pop, horizon, action_dim, generator=g)
        with torch.no_grad():
            zh = model.rollout(z0.expand(pop, -1), cand.to(device))[:, -1]
            cost = ((zh - zg) ** 2).sum(-1).cpu()
        elite = cand[cost.argsort()[:elites]]
        mu, sd = elite.mean(0), elite.std(0) + 1e-3
    return mu.numpy()


@torch.no_grad()
def plan_cost(model, z0, zg, seqs, device):
    zh = model.rollout(torch.from_numpy(z0).to(device)[None].expand(len(seqs), -1),
                       torch.from_numpy(seqs).to(device))[:, -1]
    return ((zh - torch.from_numpy(zg).to(device)[None]) ** 2).sum(-1).cpu().numpy()


def run(args):
    ev = Evaluator(args.ckpt, args.cache, args.device, args.batch_size)
    c, model, lab = ev.cache, ev.model, ev.labels
    rng = np.random.RandomState(args.seed)
    figs = os.path.join(args.out, 'figs')
    os.makedirs(figs, exist_ok=True)
    M = {'checkpoint': os.path.abspath(args.ckpt), 'name': args.name, 'tasks': c.tasks,
         'families': c.families, 'objects': c.objects, 'epoch': ev.ckpt.get('epoch'),
         'config': ev.ckpt['config']}

    # ---- encode -------------------------------------------------------------
    tr_frames = ev.episode_frames(c.split_episodes('train'))
    if len(tr_frames) > args.max_train_frames:
        tr_frames = np.sort(rng.choice(tr_frames, args.max_train_frames, replace=False))
    te_eps = c.split_episodes(args.split)
    te_frames = ev.episode_frames(te_eps)
    print(f'encoding {len(tr_frames)} train and {len(te_frames)} {args.split} frames ...', flush=True)
    ztr, zte = ev.encode(tr_frames), ev.encode(te_frames)
    row_of = -np.ones(len(c.episode_of_frame), dtype=np.int64)
    row_of[te_frames] = np.arange(len(te_frames))
    ttr, tte = ev.task_of_frame[tr_frames], ev.task_of_frame[te_frames]
    ftr, fte = ev.family_of_task[ttr], ev.family_of_task[tte]
    otr, ote = ev.object_of_task[ttr], ev.object_of_task[tte]
    np.savez_compressed(os.path.join(args.out, 'embeddings_test.npz'), z=zte, frames=te_frames, task=tte)

    # ---- A: probes from encoded latents --------------------------------------
    print('A: probes', flush=True)
    A = {'linear_r2': {}, 'linear_r2_per_dim': {}, 'mlp_r2': {}}
    probes = {}
    for q in QUANTITIES:
        f = ridge_fit(ztr, lab[q][tr_frames], args.ridge)
        probes[q] = f
        r2 = r2_score(lab[q][te_frames], f(zte))
        A['linear_r2'][q] = r2_guarded(lab[q][te_frames], f(zte), lab[q][te_frames].var(0))
        A['linear_r2_per_dim'][q] = r2.tolist()
        if not args.no_mlp and q in ('ee_pos', 'ee_rot', 'gripper', 'progress'):
            m = mlp_r2(ztr, lab[q][tr_frames], zte, lab[q][te_frames], args.seed)
            A['mlp_r2'][q] = None if m is None else float(np.mean(m))
    for key, ytr, yte, n in (('task', ttr, tte, len(c.tasks)), ('family', ftr, fte, len(c.families)),
                             ('object', otr, ote, len(c.objects))):
        A[f'{key}_acc_linear'], pred = ridge_classify(ztr, ytr, zte, yte, n, args.ridge)
        A[f'{key}_acc_knn'] = knn_classify(ztr, ytr, zte, yte, k=20)[0]
        if key == 'task':
            conf = np.zeros((n, n))
            np.add.at(conf, (yte, pred), 1)
            A['task_confusion'] = (conf / np.maximum(conf.sum(1, keepdims=True), 1)).tolist()
    # inverse-dynamics probe: action chunk from (z_t, z_{t+1})
    ok = te_frames + ev.fs <= c.starts[c.episode_of_frame[te_frames]] + c.lengths[c.episode_of_frame[te_frames]] - 1
    ok_tr = tr_frames + ev.fs <= c.starts[c.episode_of_frame[tr_frames]] + c.lengths[c.episode_of_frame[tr_frames]] - 1
    z_next_tr = ev.encode(tr_frames[ok_tr] + ev.fs)
    pair_tr = np.concatenate([ztr[ok_tr], z_next_tr], 1)
    pair_te = np.concatenate([zte[ok], zte[row_of[te_frames[ok] + ev.fs]]], 1)
    a_tr = ev.action_chunks(tr_frames[ok_tr], 1)[:, 0]
    a_te = ev.action_chunks(te_frames[ok], 1)[:, 0]
    A['action_r2_linear'] = float(ridge_r2(pair_tr, a_tr, pair_te, a_te, args.ridge).mean())
    M['A'] = A

    # ---- B: per-family probes (position vs rotation) --------------------------
    # R^2 inside a family uses the family's own variance; dimensions that barely vary
    # within the family (< 0.1% of their global variance) are skipped (NaN if all are)
    B = {}
    for fi, fam in enumerate(c.families):
        sel, sel_tr = fte == fi, ftr == fi
        B[fam] = {q: r2_guarded(lab[q][te_frames[sel]], probes[q](zte[sel]), lab[q][te_frames].var(0))
                  for q in QUANTITIES}
        # probes fitted inside the family only (is the information there at all?)
        B[fam + '_within'] = {
            q: r2_guarded(lab[q][te_frames[sel]],
                          ridge_fit(ztr[sel_tr], lab[q][tr_frames[sel_tr]], args.ridge)(zte[sel]),
                          lab[q][te_frames].var(0))
            for q in ('ee_pos', 'ee_rot', 'gripper', 'ee_lin_vel', 'ee_ang_vel')}
    M['B_family'] = B

    # ---- B: predicted latents --------------------------------------------------
    print('B: predicted latents', flush=True)
    K = args.rollout_steps
    last = c.starts[c.episode_of_frame[te_frames]] + c.lengths[c.episode_of_frame[te_frames]] - 1
    starts = te_frames[te_frames + K * ev.fs <= last][::args.rollout_stride]
    zhat = ev.rollout(zte[row_of[starts]], ev.action_chunks(starts, K))
    var = zte.var(0).mean()
    P = {'k': list(range(1, K + 1)), 'rel_err': [], 'cos': [], 'task_acc': [], 'r2': {q: [] for q in QUANTITIES},
         'r2_encoded_target': {q: [] for q in QUANTITIES}}
    task_probe = ridge_fit(ztr, np.eye(len(c.tasks))[ttr], args.ridge)
    for k in range(1, K + 1):
        tgt_frames = starts + k * ev.fs
        ztrue, zp = zte[row_of[tgt_frames]], zhat[:, k - 1]
        P['rel_err'].append(float(((zp - ztrue) ** 2).mean() / var))
        P['cos'].append(float(np.mean(np.sum(zp * ztrue, 1) / (np.linalg.norm(zp, axis=1) * np.linalg.norm(ztrue, axis=1) + 1e-8))))
        P['task_acc'].append(float((task_probe(zp).argmax(1) == ev.task_of_frame[tgt_frames]).mean()))
        for q in QUANTITIES:
            y = lab[q][tgt_frames]
            P['r2'][q].append(float(r2_score(y, probes[q](zp)).mean()))
            P['r2_encoded_target'][q].append(float(r2_score(y, probes[q](ztrue)).mean()))
    P['per_family_rel_err'] = {}
    for fi, fam in enumerate(c.families):
        sel = ev.family_of_task[ev.task_of_frame[starts]] == fi
        P['per_family_rel_err'][fam] = [float(((zhat[sel, k - 1] - zte[row_of[starts[sel] + k * ev.fs]]) ** 2).mean() / var)
                                        for k in range(1, K + 1)]
    M['B_predicted'] = P

    # ---- C: geometry -----------------------------------------------------------
    print('C: geometry', flush=True)
    G = {}
    G['effective_rank'], ev_all = effective_rank(zte)
    G['participation_ratio'] = float(ev_all.sum() ** 2 / max((ev_all ** 2).sum(), 1e-12))
    G['eigenvalues'] = (ev_all / max(ev_all.sum(), 1e-12)).tolist()
    G['effective_rank_per_task'] = {t: effective_rank(zte[tte == i])[0] for i, t in enumerate(c.tasks)}
    zc = zte - zte.mean(0)
    zn = zc / np.sqrt(zc.var(0).mean() + 1e-12)
    G['epps_pulley_raw'] = epps_pulley(zte[rng.choice(len(zte), min(4096, len(zte)), replace=False)])
    G['epps_pulley_rms_normalized'] = epps_pulley(zn[rng.choice(len(zn), min(4096, len(zn)), replace=False)])
    G['latent_mean_norm'] = float(np.linalg.norm(zte.mean(0)))
    G['latent_rms'] = float(np.sqrt((zte ** 2).mean()))
    # between- vs within-task scatter
    mu = np.stack([zte[tte == i].mean(0) for i in range(len(c.tasks))])
    within = np.mean([zte[tte == i].var(0).sum() for i in range(len(c.tasks))])
    G['between_within_task_ratio'] = float(mu.var(0).sum() / max(within, 1e-12))
    k_sub = min(args.subspace_dim, zte.shape[1])
    G['task_subspace_overlap'] = [[subspace_overlap(zte[tte == i], zte[tte == j], k_sub) for j in range(len(c.tasks))]
                                  for i in range(len(c.tasks))]
    G['subspace_dim'] = k_sub
    temporal = {t: {'straightening': [], 'r95': []} for t in c.tasks}
    for e in te_eps:
        frames = np.arange(c.starts[e], c.starts[e] + c.lengths[e], ev.fs)
        z_ep = zte[row_of[frames]]
        temporal[c.tasks[c.episode_task[e]]]['straightening'].append(straightening(z_ep))
        temporal[c.tasks[c.episode_task[e]]]['r95'].append(transition_r95(z_ep))
    G['temporal'] = {t: {k: [float(np.nanmean(v)), float(np.nanstd(v))] for k, v in d.items()}
                     for t, d in temporal.items()}
    M['C_geometry'] = G

    # ---- D: goal structure -----------------------------------------------------
    print('D: goal structure', flush=True)
    D = {'spearman_per_task': {t: [] for t in c.tasks}, 'curves': {t: [] for t in c.tasks}}
    grid = np.linspace(0, 1, 21)
    for e in te_eps:
        frames = np.arange(c.starts[e], c.starts[e] + c.lengths[e], ev.fs)
        z_ep = zte[row_of[frames]]
        d = np.linalg.norm(z_ep - z_ep[-1], axis=1)
        remaining = frames[-1] - frames
        t = c.tasks[c.episode_task[e]]
        D['spearman_per_task'][t].append(spearman(d[:-1], remaining[:-1]))
        D['curves'][t].append(np.interp(grid, (frames - frames[0]) / max(frames[-1] - frames[0], 1),
                                        d / max(d[0], 1e-12)).tolist())
    D['spearman_mean'] = {t: float(np.mean(v)) for t, v in D['spearman_per_task'].items()}
    D['curves'] = {t: np.mean(v, axis=0).tolist() for t, v in D['curves'].items()}
    D['grid'] = grid.tolist()

    # steps-to-goal probe from (z_t, z_g): log(1 + k), goal = a later frame of the same episode
    def goal_pairs(frames_pool, n, zlookup):
        out_x, out_y, out_task = [], [], []
        eps = c.episode_of_frame[frames_pool]
        pick = rng.choice(len(frames_pool), n)
        for i in pick:
            f = frames_pool[i]
            end = c.starts[eps[i]] + c.lengths[eps[i]] - 1
            kmax = (end - f) // ev.fs
            if kmax < 1:
                continue
            k = rng.randint(1, min(kmax, 60) + 1)
            zs, zg = zlookup(f), zlookup(f + k * ev.fs)
            if zs is None or zg is None:
                continue
            out_x.append(np.concatenate([zs, zg, zg - zs]))
            out_y.append(np.log1p(k))
            out_task.append(ev.task_of_frame[f])
        return np.array(out_x), np.array(out_y), np.array(out_task)
    tr_lookup = -np.ones(len(c.episode_of_frame), dtype=np.int64)
    tr_lookup[tr_frames] = np.arange(len(tr_frames))
    gx, gy, _ = goal_pairs(tr_frames, 30000, lambda f: ztr[tr_lookup[f]] if tr_lookup[f] >= 0 else None)
    hx, hy, ht = goal_pairs(te_frames, 8000, lambda f: zte[row_of[f]] if row_of[f] >= 0 else None)
    if len(gx) > 100 and len(hx) > 10:
        pred = ridge_fit(gx, gy[:, None], args.ridge)(hx)[:, 0]
        D['steps_to_goal_r2'] = float(r2_score(hy[:, None], pred[:, None])[0])
        D['steps_to_goal_spearman'] = spearman(pred, hy)
    # analogies: z(B_0) + z(A_T) - z(A_0) -> nearest test frame
    zst = Standardizer(zte)
    zs_all = zst(zte)
    an = {'same_task': [], 'same_family_other_object': []}
    eps_by_task = {i: [e for e in te_eps if c.episode_task[e] == i] for i in range(len(c.tasks))}
    for a in te_eps:
        ta = c.episode_task[a]
        for b in te_eps:
            if a == b:
                continue
            tb = c.episode_task[b]
            kind = ('same_task' if ta == tb else 'same_family_other_object'
                    if ev.family_of_task[ta] == ev.family_of_task[tb] else None)
            if kind is None:
                continue
            a0, aT = row_of[c.starts[a]], row_of[c.starts[a] + c.lengths[a] - 1]
            b0, bT = row_of[c.starts[b]], row_of[c.starts[b] + c.lengths[b] - 1]
            query = zst(zte[b0] + zte[aT] - zte[a0])
            nn = int(np.argmin(((zs_all - query) ** 2).sum(1)))
            f_nn = te_frames[nn]
            an[kind].append({
                'retrieved_progress': float(lab['progress'][f_nn, 0]),
                'retrieved_same_episode_as_B': bool(c.episode_of_frame[f_nn] == b),
                'retrieved_task_is_B': bool(ev.task_of_frame[f_nn] == tb),
                'ee_pos_error_to_B_end': float(np.linalg.norm(lab['ee_pos'][f_nn] - lab['ee_pos'][te_frames[bT]])),
                'baseline_ee_pos_error': float(np.linalg.norm(lab['ee_pos'][te_frames[b0]] - lab['ee_pos'][te_frames[bT]])),
            })
    D['analogies'] = {kind: {key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
                      for kind, rows in an.items() if rows}
    M['D_goal'] = D

    # ---- E: offline planning ---------------------------------------------------
    print('E: offline CEM planning', flush=True)
    Hp = args.plan_horizon
    cand = te_frames[te_frames + Hp * ev.fs <= last]
    picks = rng.choice(cand, min(args.plan_trials, len(cand)), replace=False)
    E = {'cost_ratio': [], 'action_nmse': [], 'first_action_nmse': [], 'gripper_agree': [], 'gt_rank': []}
    A_dim = ev.actions.shape[1] * ev.fs
    grip_cols = np.arange(6, A_dim, 7)
    for i, f in enumerate(picks):
        z0, zg = zte[row_of[f]], zte[row_of[f + Hp * ev.fs]]
        gt = ev.action_chunks(np.array([f]), Hp)[0]
        plan = cem_plan(model, z0, zg, Hp, A_dim, ev.device, args.cem_pop, args.cem_elites, args.cem_iters, seed=i)
        rand = rng.randn(256, Hp, A_dim).astype(np.float32)
        costs = plan_cost(model, z0, zg, np.concatenate([gt[None], plan[None].astype(np.float32), rand]), ev.device)
        E['cost_ratio'].append(float(costs[1] / max(costs[0], 1e-12)))
        E['gt_rank'].append(float((costs[2:] < costs[0]).mean()))
        E['action_nmse'].append(float(((plan - gt) ** 2).mean()))
        E['first_action_nmse'].append(float(((plan[0] - gt[0]) ** 2).mean()))
        E['gripper_agree'].append(float((np.sign(plan[:, grip_cols]) == np.sign(gt[:, grip_cols])).mean()))
    M['E_planning'] = {k: float(np.mean(v)) for k, v in E.items()}
    M['E_planning'].update(horizon=Hp, trials=int(len(picks)),
                           note='action_nmse in normalized action units; 1.0 ~ predicting zero (the mean)')

    with open(os.path.join(args.out, 'metrics.json'), 'w') as fh:
        json.dump(M, fh, indent=1)

    # ---- figures ---------------------------------------------------------------
    print('figures', flush=True)
    from jepa import plots
    plots.probes(M, figs)
    plots.family_probes(M, figs)
    plots.predicted(M, figs)
    plots.spectrum(M, figs)
    plots.gaussianity(zn, figs, rng)
    plots.embeddings(zte, tte, lab, te_frames, c.tasks, figs, rng, tsne=not args.no_tsne)
    plots.task_subspaces(M, figs)
    plots.temporal(M, figs)
    plots.goal_distance(M, figs)
    plots.analogies(M, figs)
    plots.planning(M, figs)
    # patch PCA on one mid-episode frame per task
    frames = np.array([c.starts[eps_by_task[i][0]] + c.lengths[eps_by_task[i][0]] // 2
                       for i in range(len(c.tasks)) if eps_by_task[i]])
    _, toks = ev.encode(frames, return_tokens=True)
    plots.patch_pca(c.images[frames], toks, [c.tasks[ev.task_of_frame[f]] for f in frames], figs)
    print(f'[done] {args.out}/metrics.json and {figs}/')
    return M


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--name', default=None, help='label in comparison plots (default: run folder name)')
    p.add_argument('--split', default='test', choices=['test', 'val'])
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-train-frames', type=int, default=40000)
    p.add_argument('--ridge', type=float, default=1e-3)
    p.add_argument('--rollout-steps', type=int, default=10, help='K for predicted-latent probes')
    p.add_argument('--rollout-stride', type=int, default=3)
    p.add_argument('--subspace-dim', type=int, default=10)
    p.add_argument('--plan-horizon', type=int, default=5, help='latent steps to the goal frame')
    p.add_argument('--plan-trials', type=int, default=100)
    p.add_argument('--cem-pop', type=int, default=300)
    p.add_argument('--cem-elites', type=int, default=30)
    p.add_argument('--cem-iters', type=int, default=10)
    p.add_argument('--no-mlp', action='store_true', help='skip MLP probes (needs scikit-learn)')
    p.add_argument('--no-tsne', action='store_true', help='skip t-SNE (needs scikit-learn)')
    args = p.parse_args()
    if args.name is None:
        args.name = os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
    run(args)


if __name__ == '__main__':
    main()
