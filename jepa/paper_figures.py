#!/usr/bin/env python
"""Publication figures comparing the latent spaces of a few world models.

    python -m jepa.paper_figures --cache cache_128 --out figures \\
        --model "MSP+SIG=runs/msp_sigreg:eval/msp_sigreg,eval/msp_sigreg_s1" \\
        --model "MSP+GCB+SIG=runs/msp_gcb_sigreg:eval/msp_gcb_sigreg" \\
        --model "MSP+SIG+SF=runs/msp_sigreg_sf_lam10:eval/msp_sigreg_sf_lam10,eval/msp_sigreg_sf10_s1"

Each --model is  LABEL=RUN_DIR:EVAL_DIR[,EVAL_DIR...]. The run directory's model_final.pt is
the model shown in the qualitative panels (with its state decoder from the first eval dir,
<eval>/state_decoder/state_decoder.pt, if present); all eval dirs (one per seed) feed the
summary bars, where each seed is drawn as a dot.

Writes (PDF + PNG) in --out:
  fig_latent_organization  t-SNE of test latents (pick-and-place coloured by EE height, rotate
                           by EE rotation) with a neighbourhood-smoothness score computed in the
                           full latent space; decoded vs measured EE state; open-loop rollouts
  fig_latent_summary       key metrics as small multiples (seeds as dots)
  fig_latent_lejepa        random 1-D projections vs N(0,1), covariance spectra, patch-token PCA
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from jepa.data import task_family  # noqa: E402
from jepa.evaluate import Evaluator  # noqa: E402
from jepa.models import torch_load  # noqa: E402

# Validated categorical slots 1-3 (all-pairs safe); continuous ramps from the same system.
MODEL_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SEQ = LinearSegmentedColormap.from_list('seq_blue', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
DIV = LinearSegmentedColormap.from_list('div_blue_red', ['#1c5cab', '#86b6ef', '#f0efec', '#ec8f8e', '#b8302f'])

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False, 'figure.facecolor': 'white',
    'savefig.facecolor': 'white', 'lines.linewidth': 2.0, 'pdf.fonttype': 42,
})


def save(fig, out, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {name}.pdf/.png')


def parse_model(spec):
    label, rest = spec.split('=', 1)
    run, evals = rest.split(':', 1)
    return {'label': label, 'run': run, 'evals': [e for e in evals.split(',') if e]}


def load_state_decoder(path, device):
    from jepa.state_decode import StateDecoder
    if not os.path.exists(path):
        return None
    ck = torch_load(path, device)
    dec = StateDecoder(ck['latent_dim'], ck['out_dim']).to(device)
    dec.load_state_dict(ck['decoder'])
    dec.eval()
    return lambda z: (np.concatenate([dec(torch.from_numpy(z[i:i + 4096]).to(device)).detach().cpu().numpy()
                                      for i in range(0, len(z), 4096)]) * ck['std'] + ck['mean'])


def knn_smoothness(z, y, k=10, n=2000, seed=0):
    """Mean |y_i - y_j| over the k nearest latent neighbours j of i, in units of std(y).
    Lower = the quantity varies smoothly over the latent space (computed in full dimension)."""
    rng = np.random.RandomState(seed)
    zs = (z - z.mean(0)) / np.maximum(z.std(0), 1e-6)
    q = rng.choice(len(z), min(n, len(z)), replace=False)
    sq = (zs ** 2).sum(1)
    d = sq[q, None] + sq[None, :] - 2 * zs[q] @ zs.T   # (n, N) squared distances
    d[np.arange(len(q)), q] = np.inf
    nn = np.argpartition(d, k, axis=1)[:, :k]
    return float(np.abs(y[q, None] - y[nn]).mean() / max(y.std(), 1e-9))


def tsne(z, seed=0):
    from sklearn.manifold import TSNE
    zs = (z - z.mean(0)) / np.maximum(z.std(0), 1e-6)
    return TSNE(2, init='pca', random_state=seed, perplexity=40).fit_transform(zs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', action='append', required=True, help='LABEL=RUN_DIR:EVAL_DIR[,EVAL_DIR...]')
    p.add_argument('--cache', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--n-tsne', type=int, default=2500, help='points per family in the t-SNE panels')
    p.add_argument('--rollout-steps', type=int, default=40)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    models = [parse_model(s) for s in a.model]
    if len(models) > 3:
        raise SystemExit('at most 3 models: the colour palette validates three series on one plot')
    rng = np.random.RandomState(a.seed)

    # ---- encode the same test frames with every model -------------------------------------
    print('encoding test frames ...', flush=True)
    for m in models:
        ev = Evaluator(os.path.join(m['run'], 'model_final.pt'), a.cache, a.device)
        m['ev'] = ev
        c = ev.cache
        m['frames'] = ev.episode_frames(c.split_episodes('test'))
        m['z'] = ev.encode(m['frames'])
        m['decode'] = load_state_decoder(os.path.join(m['evals'][0], 'state_decoder', 'state_decoder.pt'), ev.device)
    ev0 = models[0]['ev']
    c, frames, lab = ev0.cache, models[0]['frames'], ev0.labels
    fam = np.array([task_family(c.tasks[t]) for t in ev0.task_of_frame[frames]])
    height_cm = 100 * lab['ee_pos'][frames, 2]
    rot_deg = np.degrees(lab['ee_rot'][frames, 2])
    sel = {f: np.sort(rng.choice(np.flatnonzero(fam == f), min(a.n_tsne, int((fam == f).sum())), replace=False))
           for f in ('pick_and_place', 'rotate')}

    # ---- figure 1: organization ---------------------------------------------------------------
    print('figure 1: t-SNE, decoded state, rollouts ...', flush=True)
    n = len(models)
    fig = plt.figure(figsize=(3.3 * n + 0.6, 13.6))
    gs = fig.add_gridspec(5, n, height_ratios=[1, 1, 0.85, 0.85, 0.8], hspace=0.75, wspace=0.3,
                          top=0.905, bottom=0.04)
    h_lim = np.percentile(height_cm[sel['pick_and_place']], [2, 98])
    r_abs = np.percentile(np.abs(rot_deg[sel['rotate']]), 98)
    for j, m in enumerate(models):
        for r, (f, y, cmap, vlim, name) in enumerate((
                ('pick_and_place', height_cm, SEQ, h_lim, 'EE height'),
                ('rotate', rot_deg, DIV, (-r_abs, r_abs), 'EE rotation'))):
            idx = sel[f]
            emb = tsne(m['z'][idx], a.seed)
            score = knn_smoothness(m['z'][idx], y[idx])
            ax = fig.add_subplot(gs[r, j])
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=y[idx], cmap=cmap, vmin=vlim[0], vmax=vlim[1], s=4,
                            linewidths=0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f'{m["label"]}\n{f.replace("_", "-")}: neighbour spread {score:.2f}', fontsize=9,
                         color=MODEL_COLORS[j])
            if j == n - 1:
                cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
                cb.set_label(f'{name} [{"cm" if r == 0 else "deg"}]', fontsize=8)
                cb.outline.set_visible(False)
        # decoded vs measured (frozen-latent state decoder)
        ax = fig.add_subplot(gs[2, j])
        if m['decode'] is not None:
            dec = m['decode'](m['z'])
            pts = rng.choice(len(frames), min(2500, len(frames)), replace=False)
            yt, yp = height_cm[pts], 100 * dec[pts, 2]
            lo, hi = np.percentile(np.r_[yt, yp], [1, 99])
            ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls='--')
            ax.scatter(yt, yp, s=4, color=MODEL_COLORS[j], alpha=0.35, linewidths=0)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            rmse = float(np.sqrt(((100 * dec[:, 0:3] - 100 * lab['ee_pos'][frames]) ** 2).sum(1).mean()))
            ax.set_title(f'{m["label"]}: EE height decoded from z\n(3-D position RMSE {rmse:.1f} cm)', fontsize=9)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, f'{m["label"]}: no state decoder\n(run jepa.state_decode)', ha='center',
                    va='center', transform=ax.transAxes, color=INK2)
        if m['decode'] is not None:
            ax.set_xlabel('measured EE height [cm]')
        if j == 0 and m['decode'] is not None:
            ax.set_ylabel('decoded EE height [cm]')
        ax = fig.add_subplot(gs[3, j])
        if m['decode'] is None:
            ax.axis('off')
        if m['decode'] is not None:
            pts = rng.choice(len(frames), min(2500, len(frames)), replace=False)
            yt, yp = rot_deg[pts], np.degrees(dec[pts, 5])
            lim = np.percentile(np.abs(np.r_[yt, yp]), 99)
            ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=1, ls='--')
            ax.scatter(yt, yp, s=4, color=MODEL_COLORS[j], alpha=0.35, linewidths=0)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            err = float(np.sqrt(((yp - yt) ** 2).mean()))
            ax.set_title(f'{m["label"]}: EE rotation decoded from z\n(RMSE {err:.1f} deg)', fontsize=9)
            ax.set_xlabel('measured EE rotation [deg]')
            if j == 0:
                ax.set_ylabel('decoded EE rotation [deg]')

    # open-loop rollouts: one rotate and one pick-and-place test episode, all models on one axis
    sub = gs[4, :].subgridspec(1, 2, wspace=0.25)
    for col, (task_prefix, idx_q, name, unit, scale) in enumerate((
            ('rotate', 5, 'EE rotation', 'deg', 180 / np.pi),
            ('pick_and_place', 2, 'EE height', 'cm', 100.0))):
        ax = fig.add_subplot(sub[0, col])
        eps = [e for e in c.split_episodes('test') if c.tasks[c.episode_task[e]].startswith(task_prefix)]
        e = max(eps, key=lambda e: c.lengths[e])  # the longest test episode of the family
        s0 = int(c.starts[e])
        steps = min(a.rollout_steps, (int(c.lengths[e]) - 1) // ev0.fs)
        idx = s0 + np.arange(steps + 1) * ev0.fs
        t = np.arange(steps + 1) * ev0.fs / 10.0
        meas = lab['ee_rot'][idx, 2] if idx_q == 5 else lab['ee_pos'][idx, 2]
        ax.plot(t, meas * scale, color=INK, lw=2.4, label='measured')
        for j, m in enumerate(models):
            if m['decode'] is None:
                continue
            evm = m['ev']
            z0 = evm.encode(idx[:1])
            zh = evm.rollout(z0, evm.action_chunks(idx[:1], steps))[0]
            pred = m['decode'](np.concatenate([z0, zh]))[:, idx_q]
            ax.plot(t, pred * scale, color=MODEL_COLORS[j], lw=1.6, label=m['label'])
        ax.set_xlabel('time [s] (open loop, recorded actions)')
        ax.set_ylabel(f'{name} [{unit}]')
        ax.set_title(f'{c.tasks[c.episode_task[e]]}: predicted vs measured {name}', fontsize=9)
        ax.legend(fontsize=8, loc='best')
    fig.suptitle('Latent organization: t-SNE (neighbour spread in full latent space, lower = smoother), '
                 'decoded robot state, and open-loop predictions', fontsize=10.5, y=0.975)
    save(fig, a.out, 'fig_latent_organization')

    # ---- figure 2: summary small multiples --------------------------------------------------
    print('figure 2: summary metrics ...', flush=True)

    def get(ed, path):
        file, *keys = path
        fp = os.path.join(ed, file)
        if not os.path.exists(fp):
            return np.nan
        with open(fp) as fh:
            v = json.load(fh)
        try:  # older evaluations may lack a metric: show a gap rather than fail
            for k in keys:
                v = v[k]
            return float(v)
        except (KeyError, IndexError, TypeError):
            return np.nan

    metrics = [
        ('Family balance', 'weaker / stronger family, pos+rot R² (↑)', ('metrics.json', 'C_organization', 'family_balance')),
        ('Latent ↔ physical distance', 'Spearman within task (↑)', ('metrics.json', 'C_organization', 'latent_physical_distance_spearman')),
        ('EE rotation on rotate tasks', 'probe R² (↑)', ('metrics.json', 'B_family', 'rotate_within', 'ee_rot')),
        ('Task identity', 'kNN accuracy (↑)', ('metrics.json', 'A', 'task_acc_knn')),
        ('EE position, 2 s prediction', 'error [cm] (↓)', ('state_decoder/state_decoder_metrics.json', 'error', 'all', 'pred', 'ee_pos_cm', -1)),
        ('EE rotation, 2 s prediction', 'error [deg] (↓)', ('state_decoder/state_decoder_metrics.json', 'error', 'all', 'pred', 'ee_rot_deg', -1)),
        ('Distance to goal', 'Spearman with remaining time (↑)', None),
        ('Effective rank', 'of the latent covariance (↑)', ('metrics.json', 'C_geometry', 'effective_rank')),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 5.6))
    for ax, (title, ylabel, path) in zip(axes.flat, metrics):
        for j, m in enumerate(models):
            if path is None:
                vals = []
                for ed in m['evals']:
                    fp = os.path.join(ed, 'metrics.json')
                    if os.path.exists(fp):
                        with open(fp) as fh:
                            vals.append(float(np.mean(list(json.load(fh)['D_goal']['spearman_mean'].values()))))
                vals = np.array(vals) if vals else np.array([np.nan])
            else:
                vals = np.array([get(ed, path) for ed in m['evals']])
            if np.all(np.isnan(vals)):
                ax.text(j, 0, 'n/a', ha='center', va='bottom', fontsize=8, color=INK2)
                continue
            vals = vals[~np.isnan(vals)]
            mean = float(np.mean(vals))
            ax.bar(j, mean, 0.62, color=MODEL_COLORS[j], zorder=2)
            ax.scatter(np.full(len(vals), j) + np.linspace(-0.12, 0.12, len(vals)) * (len(vals) > 1), vals,
                       s=22, color='white', edgecolors=INK, linewidths=0.9, zorder=3)
            ax.annotate(f'{mean:.2f}', (j + 0.33, mean), xytext=(2, 0), textcoords='offset points',
                        ha='left', va='center', fontsize=8, color=INK, zorder=4)
        ax.set_xticks(range(n), [m['label'] for m in models], rotation=15, fontsize=8)
        better = 'higher is better' if ylabel.endswith('(↑)') else 'lower is better'
        ax.set_title(f'{title}\n({better})', fontsize=9.5)
        ax.set_ylabel(ylabel.replace(' (↑)', '').replace(' (↓)', ''), fontsize=8, color=INK2)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlim(-0.6, n - 0.25)
        ax.grid(axis='x', visible=False)
        ax.set_axisbelow(True)
    fig.suptitle('What each objective gives the latent space (bars: mean over seeds; dots: individual seeds)',
                 fontsize=10.5)
    fig.tight_layout()
    save(fig, a.out, 'fig_latent_summary')

    # ---- figure 3: LeJEPA-style ------------------------------------------------------------------
    print('figure 3: projections, spectra, patch PCA ...', flush=True)
    shown = []  # mid-episode frames of four different tasks (two per family)
    for prefix in ('pick_and_place', 'rotate'):
        tids = [i for i, t in enumerate(c.tasks) if t.startswith(prefix)][:2]
        for ti in tids:
            eps = [e for e in c.split_episodes('test') if c.episode_task[e] == ti]
            if eps:
                shown.append(int(c.starts[eps[0]] + c.lengths[eps[0]] // 2))
    shown = np.array(shown)
    fig = plt.figure(figsize=(14, 2.6 * n + 2.4))
    gs = fig.add_gridspec(n + 1, 7, height_ratios=[1] * n + [0.9], wspace=0.3, hspace=0.5, top=0.92)
    xs = np.linspace(-4, 4, 200)
    dirs = rng.randn(models[0]['z'].shape[1], 3)
    dirs /= np.linalg.norm(dirs, axis=0)
    ax_spec = fig.add_subplot(gs[n, 0:3])
    for j, m in enumerate(models):
        z = m['z']
        zc = z - z.mean(0)
        zn = zc / np.sqrt(zc.var(0).mean() + 1e-12)
        for k in range(3):
            ax = fig.add_subplot(gs[j, k])
            ax.hist(zn @ dirs[:, k], bins=60, range=(-4, 4), density=True, color=MODEL_COLORS[j], alpha=0.75,
                    edgecolor='white', linewidth=0.3)
            ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=1.2)
            ax.set_yticks([])
            ax.grid(False)
            if k == 0:
                ax.set_ylabel(m['label'], color=MODEL_COLORS[j], fontsize=9.5, fontweight='bold')
            if j == 0:
                ax.set_title(f'1-D projection {k + 1}' + (' (black: N(0,1))' if k == 0 else ''), fontsize=8.5)
        ev_ = np.clip(np.linalg.eigvalsh(np.cov(zn, rowvar=False))[::-1], 1e-8, None)
        ax_spec.semilogy(np.arange(1, len(ev_) + 1), ev_ / ev_.sum(), color=MODEL_COLORS[j], label=m['label'])
        # patch-token PCA as RGB (one PCA per model, shared across its frames)
        _, toks = m['ev'].encode(shown, return_tokens=True)
        nt, pt, ct = toks.shape
        g = int(round(np.sqrt(pt)))
        flat = toks.reshape(-1, ct)
        flat = flat - flat.mean(0)
        proj = flat @ np.linalg.svd(flat, full_matrices=False)[2][:3].T
        lo, hi = np.percentile(proj, 1, axis=0), np.percentile(proj, 99, axis=0)
        rgb = np.clip((proj - lo) / np.maximum(hi - lo, 1e-8), 0, 1).reshape(nt, g, g, 3)
        for k in range(min(4, nt)):
            ax = fig.add_subplot(gs[j, 3 + k])
            ax.imshow(rgb[k], interpolation='nearest')
            ax.axis('off')
            if j == 0:
                ax.set_title(c.tasks[ev0.task_of_frame[shown[k]]].replace('pick_and_place', 'pick-place'),
                             fontsize=8)
    ax_spec.set_xlabel('principal component')
    ax_spec.set_ylabel('fraction of variance')
    ax_spec.set_title('covariance spectrum (flatter = more dimensions used)', fontsize=9)
    ax_spec.legend(fontsize=8)
    for k in range(min(4, len(shown))):
        ax = fig.add_subplot(gs[n, 3 + k])
        ax.imshow(c.images[shown[k]])
        ax.axis('off')
        if k == 0:
            ax.set_title('input frames (patch PCA above)', fontsize=8, loc='left')
    fig.suptitle('Gaussianity, dimensionality and what the encoder attends to (LeJEPA-style)', fontsize=10.5)
    save(fig, a.out, 'fig_latent_lejepa')
    print(f'[done] {a.out}')


if __name__ == '__main__':
    main()
