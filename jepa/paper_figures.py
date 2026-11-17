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
  fig_latent_geometry      is the latent a map of the robot's physical state? nearest-neighbour
                           retrieval across episodes (images, with their physical offsets),
                           neighbour error vs k, and physical distance vs latent-distance rank
  fig_latent_trajectories  test episodes of one task in the 2 leading PCs of that task's latents,
                           coloured by the task's physical variable

--figures selects a subset, e.g. --figures geometry,trajectories (skips the slow t-SNE).
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


def short_task(t):
    return t.replace('pick_and_place_', 'pick ').replace('rotate_', 'rotate ')


def _sqdist(q, z):
    return np.maximum((q ** 2).sum(1)[:, None] + (z ** 2).sum(1)[None] - 2 * q @ z.T, 0)


def _test_context(models):
    ev0 = models[0]['ev']
    c, frames = ev0.cache, models[0]['frames']
    task = ev0.task_of_frame[frames]
    return {'ev0': ev0, 'c': c, 'frames': frames, 'ep': c.episode_of_frame[frames], 'task': task,
            'fam': np.array([task_family(c.tasks[t]) for t in task]),
            'pos': 100 * ev0.labels['ee_pos'][frames],                 # cm
            'rot': np.degrees(ev0.labels['ee_rot'][frames]),           # rotation vector, deg
            'progress': ev0.labels['progress'][frames, 0]}


def _default_queries(ctx):
    """One frame per family, about halfway through the median-length test episode of that family."""
    c, out = ctx['c'], []
    for f in ('pick_and_place', 'rotate'):
        eps = np.unique(ctx['ep'][ctx['fam'] == f])
        if not len(eps):
            continue
        e = sorted(eps, key=lambda e: c.lengths[e])[len(eps) // 2]
        idx = np.flatnonzero(ctx['ep'] == e)
        out.append(int(idx[np.argmin(np.abs(ctx['progress'][idx] - 0.55))]))
    return out


def geometry_figure(models, a, rng):
    """Is the latent a map of the robot's physical state? Nearest neighbours in latent space, taken
    from *other* test episodes, should show the robot in the same configuration."""
    print('figure: latent geometry (retrieval, neighbour error, distance rank) ...', flush=True)
    ctx = _test_context(models)
    c, frames, ep, fam, pos, rot = ctx['c'], ctx['frames'], ctx['ep'], ctx['fam'], ctx['pos'], ctx['rot']
    n, n_nb = len(models), 5
    if a.queries:
        lookup = {int(f): i for i, f in enumerate(frames)}
        qs = [lookup[int(f)] for f in a.queries.split(',') if int(f) in lookup]
    else:
        qs = _default_queries(ctx)

    fig = plt.figure(figsize=(14, 1.45 * n * ((len(qs) + 1) // 2) + 3.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.45 * n * ((len(qs) + 1) // 2), 3.0], hspace=0.32,
                             top=0.93, bottom=0.07)
    ret = outer[0].subgridspec((len(qs) + 1) // 2, min(2, len(qs)), wspace=0.08, hspace=0.25)
    for qi, q in enumerate(qs):
        blk = ret[qi // 2, qi % 2].subgridspec(n, n_nb + 2, wspace=0.06, hspace=0.42,
                                               width_ratios=[1, 0.22] + [1] * n_nb)
        axq = fig.add_subplot(blk[:, 0])
        axq.imshow(c.images[frames[q]])
        axq.set_xticks([])
        axq.set_yticks([])
        for sp in axq.spines.values():
            sp.set_visible(True)
            sp.set_color(INK)
            sp.set_linewidth(2)
        axq.set_title(f'query\n{short_task(c.tasks[ctx["task"][q]])}', fontsize=8, fontweight='bold')
        for j, m in enumerate(models):
            d = _sqdist(m['z'][q:q + 1], m['z'])[0]
            d[ep == ep[q]] = np.inf
            picks, seen = [], set()
            for i in np.argsort(d):  # the closest frame in each of the closest other episodes
                if ep[i] in seen or not np.isfinite(d[i]):
                    continue
                seen.add(ep[i])
                picks.append(i)
                if len(picks) == n_nb:
                    break
            for k, i in enumerate(picks):
                ax = fig.add_subplot(blk[j, k + 2])
                ax.imshow(c.images[frames[i]])
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_color(MODEL_COLORS[j])
                    sp.set_linewidth(1.5)
                dp = np.linalg.norm(pos[i] - pos[q])
                dr = np.linalg.norm(rot[i] - rot[q])
                same = ctx['task'][i] == ctx['task'][q]
                ax.set_title(f'{short_task(c.tasks[ctx["task"][i]])}{"" if same else " ✗"}\n'
                             f'Δ{dp:.1f} cm  Δ{dr:.0f}°', fontsize=6.5, color=INK if same else '#b8302f', pad=2)
                if k == 0:
                    ax.set_ylabel(m['label'], color=MODEL_COLORS[j], fontsize=8.5, fontweight='bold')

    # ---- quantitative curves ----------------------------------------------------------------
    curves = outer[1].subgridspec(1, 4, wspace=0.38)
    K = 50
    specs = (('pick_and_place', 'position', lambda i, j: np.linalg.norm(pos[i] - pos[j], axis=-1), 'cm'),
             ('rotate', 'rotation', lambda i, j: np.linalg.norm(rot[i] - rot[j], axis=-1), 'deg'))
    for col, (f, name, dist, unit) in enumerate(specs):
        idx = np.flatnonzero(fam == f)
        if not len(idx):
            continue
        qidx = rng.choice(idx, min(1000, len(idx)), replace=False)
        ax = fig.add_subplot(curves[0, col])
        for j, m in enumerate(models):
            D = _sqdist(m['z'][qidx], m['z'])
            D[ep[qidx][:, None] == ep[None, :]] = np.inf
            nn = np.argpartition(D, K, axis=1)[:, :K]
            nn = np.take_along_axis(nn, np.argsort(np.take_along_axis(D, nn, 1), 1), 1)
            err = dist(qidx[:, None], nn)                       # (Q, K)
            cum = np.cumsum(err, 1) / np.arange(1, K + 1)
            ax.plot(np.arange(1, K + 1), cum.mean(0), color=MODEL_COLORS[j], label=m['label'])
        rnd = dist(qidx, rng.choice(len(frames), len(qidx)))
        ax.axhline(rnd.mean(), color=INK2, lw=1, ls='--')
        ax.text(K, rnd.mean(), 'random frame', ha='right', va='top', fontsize=7, color=INK2)
        ax.set_xscale('log')
        ax.set_ylim(bottom=0)
        ax.set_xlabel('k nearest latent neighbours (other episodes)')
        ax.set_ylabel(f'EE {name} difference [{unit}]')
        ax.set_title(f'{f.replace("_", "-")}: physical gap of\nlatent neighbours (lower is better)', fontsize=9)
        if col == 0:
            ax.legend(fontsize=7.5, loc='center right')
    for col, (f, name, dist, unit) in enumerate(specs):
        ax = fig.add_subplot(curves[0, 2 + col])
        idx = np.flatnonzero(fam == f)
        if not len(idx):
            continue
        i = rng.choice(idx, 30000)
        j_ = rng.choice(idx, 30000)
        keep = (ctx['task'][i] == ctx['task'][j_]) & (ep[i] != ep[j_])  # same task, different episode
        i, j_ = i[keep], j_[keep]
        phys = dist(i, j_)
        edges = np.linspace(0, 100, 21)
        mid = 0.5 * (edges[1:] + edges[:-1])
        for jm, m in enumerate(models):
            dl = np.linalg.norm(m['z'][i] - m['z'][j_], axis=1)
            rank = 100 * np.argsort(np.argsort(dl)) / max(len(dl) - 1, 1)
            b = np.clip(np.digitize(rank, edges) - 1, 0, len(mid) - 1)
            q = np.array([np.percentile(phys[b == k], [25, 50, 75]) if (b == k).any() else [np.nan] * 3
                          for k in range(len(mid))])
            rho = float(np.corrcoef(rank, np.argsort(np.argsort(phys)))[0, 1])
            ax.fill_between(mid, q[:, 0], q[:, 2], color=MODEL_COLORS[jm], alpha=0.15, linewidth=0)
            ax.plot(mid, q[:, 1], color=MODEL_COLORS[jm], label=f'{m["label"]} (ρ={rho:.2f})')
        ax.set_ylim(bottom=0)
        ax.set_xlabel('latent distance (percentile among pairs)')
        ax.set_ylabel(f'EE {name} difference [{unit}]')
        ax.set_title(f'{f.replace("_", "-")}: physical vs latent distance\n(same task, different episodes)',
                     fontsize=9)
        ax.legend(fontsize=7, loc='upper left')
    fig.suptitle('Is the latent a map of the robot\'s state?  Top: nearest test frames in latent space from other '
                 'episodes (Δ = physical offset from the query; ✗ = different task).  Bottom: the same, measured.',
                 fontsize=10.5, y=0.985)
    save(fig, a.out, 'fig_latent_geometry')


def _trajectory_rows(ctx, a):
    c = ctx['c']
    rows = []
    for prefix in ('pick_and_place', 'rotate'):
        tids = [t for t in np.unique(ctx['task']) if c.tasks[t].startswith(prefix)]
        if a.trajectory_tasks:
            tids = [t for t in tids if c.tasks[t] in a.trajectory_tasks.split(',')] or tids
        if not tids:
            continue
        t = tids[0]
        if prefix == 'rotate':  # rotation about the vertical axis; R² of that angle
            rows.append((t, 'EE rotation', 'deg', ctx['rot'][:, 2], ctx['rot'][:, 2:3], 'EE rotation'))
        else:  # colour by the EE axis that varies most within the task; R² of the 3-D position
            ax_i = int(np.argmax(ctx['pos'][ctx['task'] == t].var(0)))
            rows.append((t, f'EE {"xyz"[ax_i]}', 'cm', ctx['pos'][:, ax_i], ctx['pos'], '3-D EE position'))
    return rows


def trajectories_figure(models, a):
    """Test episodes of one task in the two leading principal components of that task's latents.
    If the main directions of variation are physical, the colour changes smoothly along one axis and
    the episodes trace the same path."""
    print('figure: per-task latent trajectories ...', flush=True)
    ctx = _test_context(models)
    c, ep, n = ctx['c'], ctx['ep'], len(models)
    rows = _trajectory_rows(ctx, a)
    fig, axes = plt.subplots(len(rows), n + 1, figsize=(3.6 * n + 4.2, 3.4 * len(rows)), squeeze=False,
                             gridspec_kw={'width_ratios': [1] * n + [0.9]})
    ks = np.array([1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 192, 256, 384, 512])
    ev0, lab = ctx['ev0'], ctx['ev0'].labels

    def fit(F, Yt, F_te=None, Y_te=None):
        X = np.c_[F, np.ones(len(F))]
        w = np.linalg.lstsq(X, Yt, rcond=None)[0]
        if F_te is not None:  # evaluate on held-out frames
            X, Yt = np.c_[F_te, np.ones(len(F_te))], Y_te
        res = Yt - X @ w
        r2 = 1 - res.var(0).sum() / max(Yt.var(0).sum(), 1e-12)  # fraction of variance explained
        return r2, float(np.sqrt((res ** 2).sum(1).mean()))       # RMS error (3-D norm for position)

    def targets(frames_, unit, Y_cols):
        if unit == 'deg':
            return np.degrees(lab['ee_rot'][frames_])[:, 2:3]
        return 100 * lab['ee_pos'][frames_]

    record = {}

    for r, (t, name, unit, y, Y, r2_name) in enumerate(rows):
        idx = np.flatnonzero(ctx['task'] == t)
        yt, Yt = y[idx], Y[idx]
        if unit == 'deg':
            lim = np.percentile(np.abs(yt), 98)
            cmap, vmin, vmax = DIV, -lim, lim
        else:
            cmap, (vmin, vmax) = SEQ, np.percentile(yt, [2, 98])
        for j, m in enumerate(models):
            z = m['z'][idx] - m['z'][idx].mean(0)
            vt = np.linalg.svd(z, full_matrices=False)[2]
            pc = z @ vt[:2].T
            r2 = fit(pc, Yt)[0]
            # error curve: PCA and linear fit on the task's *training* episodes, error on its test episodes
            tr_frames = ev0.episode_frames([e for e in c.split_episodes('train') if c.episode_task[e] == t])
            z_tr = m['ev'].encode(tr_frames)
            mu = z_tr.mean(0)
            vt_tr = np.linalg.svd(z_tr - mu, full_matrices=False)[2]
            Y_tr = targets(tr_frames, unit, None)
            kk = ks[ks <= vt_tr.shape[0]]
            fits = [fit((z_tr - mu) @ vt_tr[:k].T, Y_tr, (m['z'][idx] - mu) @ vt_tr[:k].T, Yt) for k in kk]
            fits_in = [fit((z_tr - mu) @ vt_tr[:k].T, Y_tr) for k in kk]
            axes[r, n].plot(kk, [f[1] for f in fits], color=MODEL_COLORS[j], marker='o', ms=3, label=m['label'])
            axes[r, n].plot(kk, [f[1] for f in fits_in], color=MODEL_COLORS[j], lw=1, ls=':', alpha=0.8)
            record.setdefault(c.tasks[t], {})[m['label']] = {
                'k': kk.tolist(), 'r2_test': [f[0] for f in fits], f'rmse_{unit}_test': [f[1] for f in fits],
                'r2_train': [f[0] for f in fits_in], f'rmse_{unit}_train': [f[1] for f in fits_in],
                'n_train_frames': int(len(tr_frames)), 'n_test_frames': int(len(idx))}
            ax = axes[r, j]
            for e in np.unique(ep[idx]):
                k = ep[idx] == e
                ax.plot(pc[k, 0], pc[k, 1], color=INK2, lw=0.5, alpha=0.5, zorder=1)
                ax.scatter(pc[k, 0][:1], pc[k, 1][:1], s=14, color=INK, marker='o', zorder=3, linewidths=0)
            sc = ax.scatter(pc[:, 0], pc[:, 1], c=yt, cmap=cmap, vmin=vmin, vmax=vmax, s=6, linewidths=0, zorder=2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'{m["label"]}\n{r2_name} explained by PC1-2: R²={r2:.2f}', fontsize=9, color=MODEL_COLORS[j])
            if j == 0:
                ax.set_ylabel(f'{c.tasks[t]}\nPC2', fontsize=8.5)
            ax.set_xlabel('PC1', fontsize=8)
            if j == n - 1:
                cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
                cb.set_label(f'{name} [{unit}]', fontsize=8)
                cb.outline.set_visible(False)
        ax = axes[r, n]
        base = float(np.sqrt(((Yt - Yt.mean(0)) ** 2).sum(1).mean()))  # error of predicting the mean
        ax.axhline(base, color=INK2, lw=1, ls='--')
        ax.set_xscale('log')
        ax.set_yscale('log')
        from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        shown_k = [k for k in (1, 2, 5, 12, 30, 80, 192, 512) if k <= kk.max()]
        ax.set_xticks(shown_k, [str(k) for k in shown_k])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('number of leading PCs (fit on train episodes)', fontsize=8)
        ax.set_ylabel(f'{r2_name} error [{unit}]', fontsize=8)
        ax.set_title(f'{r2_name} left unexplained by the\nleading principal directions (lower is better)',
                     fontsize=9)
        from matplotlib.lines import Line2D
        style = [Line2D([], [], color=INK, lw=2, marker='o', ms=3, label='held-out test episodes'),
                 Line2D([], [], color=INK, lw=1, ls=':', label='train episodes (used for the fit)'),
                 Line2D([], [], color=INK2, lw=1, ls='--', label='no information (predict the mean)')]
        models_leg = [Line2D([], [], color=MODEL_COLORS[j], lw=2, label=m['label']) for j, m in enumerate(models)]
        ax.legend(handles=(models_leg if r == 0 else style), fontsize=7, loc='lower left')
    fig.suptitle('Test episodes in the two leading principal directions of one task\'s latents '
                 '(black dot = episode start; colour = physical state)', fontsize=10.5)
    fig.tight_layout(w_pad=3.0)
    save(fig, a.out, 'fig_latent_trajectories')
    with open(os.path.join(a.out, 'latent_trajectories_pcs.json'), 'w') as fh:
        json.dump(record, fh, indent=1)


def pc_error_figure(models, a):
    """EE position error (pick-and-place task) and EE rotation error (rotate task) of a linear readout
    from the k leading principal directions of the task's latents, one row per object. PCA and the
    fit use the task's training episodes; the error is measured on its validation + test episodes."""
    print('figure: error vs number of principal directions ...', flush=True)
    ctx = _test_context(models)
    c, ev0 = ctx['c'], ctx['ev0']
    lab = ev0.labels
    ks = np.array([1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100, 120, 140, 160, 192, 256, 384, 512])
    objects = [t[len('rotate_'):] for t in c.tasks if t.startswith('rotate_')
               and f'pick_and_place_{t[len("rotate_"):]}' in c.tasks]
    cols = (('pick_and_place', '3-D EE position', 'cm'), ('rotate', 'EE rotation', 'deg'))
    fig, axes = plt.subplots(len(objects), 2, figsize=(9.6, 3.3 * len(objects)), squeeze=False)
    record = {}
    for r, obj in enumerate(objects):
        for col, (fam_, qname, unit) in enumerate(cols):
            ax = axes[r, col]
            t = c.tasks.index(f'{fam_}_{obj}')
            eps_of = lambda split: [e for e in c.split_episodes(split) if c.episode_task[e] == t]
            tr = ev0.episode_frames(eps_of('train'))
            ho_eps = eps_of('val') + eps_of('test')
            ho = ev0.episode_frames(ho_eps)

            def targets(f):
                return np.degrees(lab['ee_rot'][f])[:, 2:3] if unit == 'deg' else 100 * lab['ee_pos'][f]
            Y_tr, Y_ho = targets(tr), targets(ho)
            for j, m in enumerate(models):
                z_tr, z_ho = m['ev'].encode(tr), m['ev'].encode(ho)
                mu = z_tr.mean(0)
                vt = np.linalg.svd(z_tr - mu, full_matrices=False)[2]
                kk = ks[ks <= vt.shape[0]]
                err, r2 = [], []
                for k in kk:
                    X_tr = np.c_[(z_tr - mu) @ vt[:k].T, np.ones(len(tr))]
                    X_ho = np.c_[(z_ho - mu) @ vt[:k].T, np.ones(len(ho))]
                    res = Y_ho - X_ho @ np.linalg.lstsq(X_tr, Y_tr, rcond=None)[0]
                    err.append(float(np.sqrt((res ** 2).sum(1).mean())))
                    r2.append(float(1 - res.var(0).sum() / max(Y_ho.var(0).sum(), 1e-12)))
                ax.plot(kk, err, color=MODEL_COLORS[j], marker='o', ms=2.5, label=m['label'])
                record.setdefault(c.tasks[t], {})[m['label']] = {
                    'k': kk.tolist(), f'rmse_{unit}': err, 'r2': r2,
                    'n_train_episodes': len(eps_of('train')), 'n_heldout_episodes': len(ho_eps)}
            ax.set_yscale('log')  # semilogy: linear k, log error
            from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(0, kk.max() + 2)
            if r == len(objects) - 1:
                ax.set_xlabel('number of leading principal directions')
            ax.set_ylabel(f'{qname} error [{unit}]')
            ax.set_title(f'{c.tasks[t]} ({len(ho_eps)} held-out episodes)', fontsize=9.5)
            if r == 0 and col == 1:
                ax.legend(fontsize=8, loc='upper right')
    for col in range(2):  # same y-range down each column, so objects compare directly
        lo = min(ax.get_ylim()[0] for ax in axes[:, col])
        hi = max(ax.get_ylim()[1] for ax in axes[:, col])
        for ax in axes[:, col]:
            ax.set_ylim(lo, hi)
    fig.suptitle('How much of the robot state a linear readout of the leading latent directions recovers '
                 '(lower is better)', fontsize=10.5)
    fig.tight_layout()
    save(fig, a.out, 'fig_latent_pc_error')
    with open(os.path.join(a.out, 'latent_pc_error.json'), 'w') as fh:
        json.dump(record, fh, indent=1)


def pc_content_figure(models, a):
    """What do the leading principal directions of one task's latents encode? Each cell is the
    fraction of a PC's variance explained by one variable: squared correlation for a physical
    quantity, and the between-episode share of variance for 'episode' (anything constant within an
    episode: object placement, lighting, appearance)."""
    print('figure: content of the leading principal directions ...', flush=True)
    ctx = _test_context(models)
    c, ep, n, npc = ctx['c'], ctx['ep'], len(models), 5
    lab, frames = ctx['ev0'].labels, ctx['frames']
    speed = 100 * np.linalg.norm(lab['ee_lin_vel'][frames], axis=1)
    cand = [('EE x', ctx['pos'][:, 0]), ('EE y', ctx['pos'][:, 1]), ('EE z', ctx['pos'][:, 2]),
            ('EE rotation', ctx['rot'][:, 2]), ('gripper', lab['gripper'][frames, 0]),
            ('progress', ctx['progress']), ('EE speed', speed), ('episode', None)]
    rows = _trajectory_rows(ctx, a)
    fig, axes = plt.subplots(len(rows), n, figsize=(3.3 * n + 1.0, 2.9 * len(rows)), squeeze=False)
    record = {}
    for r, row in enumerate(rows):
        t = row[0]
        idx = np.flatnonzero(ctx['task'] == t)
        for j, m in enumerate(models):
            z = m['z'][idx] - m['z'][idx].mean(0)
            _, sv, vt = np.linalg.svd(z, full_matrices=False)
            frac = sv ** 2 / (sv ** 2).sum()
            pcs = z @ vt[:npc].T
            M = np.zeros((npc, len(cand)))
            for i in range(npc):
                p_ = pcs[:, i]
                for k, (name, v) in enumerate(cand):
                    if v is None:
                        e = ep[idx]
                        means = np.array([p_[e == u].mean() for u in np.unique(e)])
                        counts = np.array([(e == u).sum() for u in np.unique(e)])
                        M[i, k] = (counts * (means - p_.mean()) ** 2).sum() / max(((p_ - p_.mean()) ** 2).sum(), 1e-12)
                    else:
                        vv = v[idx]
                        M[i, k] = 0.0 if vv.std() < 1e-9 else np.corrcoef(p_, vv)[0, 1] ** 2
            ax = axes[r, j]
            im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect='auto')
            for i in range(npc):
                for k in range(len(cand)):
                    ax.text(k, i, f'{M[i, k]:.2f}', ha='center', va='center', fontsize=6.5,
                            color='white' if M[i, k] > 0.55 else INK)
            ax.set_xticks(range(len(cand)), [nm for nm, _ in cand], rotation=45, ha='right', fontsize=7.5)
            ax.set_yticks(range(npc), [f'PC{i + 1} ({100 * frac[i]:.0f}%)' for i in range(npc)], fontsize=7.5)
            ax.grid(False)
            ax.set_title(f'{m["label"]}' + f'\n{c.tasks[t]}', fontsize=9, color=MODEL_COLORS[j])
            record.setdefault(c.tasks[t], {})[m['label']] = {
                'variance_fraction': frac[:npc].tolist(),
                'r2': {nm: M[:, k].tolist() for k, (nm, _) in enumerate(cand)}}
        cb = fig.colorbar(im, ax=axes[r, n - 1], fraction=0.046, pad=0.03)
        cb.set_label('fraction of PC variance explained', fontsize=7.5)
        cb.outline.set_visible(False)
    fig.suptitle('What the leading principal directions of one task\'s latents encode '
                 '(in brackets: share of latent variance; "episode" = constant within an episode)', fontsize=10)
    fig.tight_layout()
    save(fig, a.out, 'fig_latent_pc_content')
    with open(os.path.join(a.out, 'latent_pc_content.json'), 'w') as fh:
        json.dump(record, fh, indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', action='append', required=True, help='LABEL=RUN_DIR:EVAL_DIR[,EVAL_DIR...]')
    p.add_argument('--cache', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--n-tsne', type=int, default=2500, help='points per family in the t-SNE panels')
    p.add_argument('--rollout-steps', type=int, default=40)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--figures', default='organization,summary,lejepa,geometry,trajectories,pc_content,pc_error')
    p.add_argument('--queries', default='', help='comma-separated global frame indices for the retrieval '
                   'panel (default: one mid-episode frame per family)')
    p.add_argument('--trajectory-tasks', default='', help='comma-separated task names for fig_latent_trajectories')
    a = p.parse_args()
    figs = set(a.figures.split(','))
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

    n = len(models)
    if 'geometry' in figs:
        geometry_figure(models, a, rng)
    if 'trajectories' in figs:
        trajectories_figure(models, a)
    if 'pc_error' in figs:
        pc_error_figure(models, a)
    if 'pc_content' in figs:
        pc_content_figure(models, a)
    if 'organization' in figs:
        organization_figure(models, a, rng, sel, height_cm, rot_deg, frames, lab, c, ev0)
    if 'summary' in figs:
        summary_figure(models, a)
    if 'lejepa' in figs:
        lejepa_figure(models, a, rng, c, ev0)
    print(f'[done] {a.out}')


def organization_figure(models, a, rng, sel, height_cm, rot_deg, frames, lab, c, ev0):
    print('figure: t-SNE, decoded state, rollouts ...', flush=True)
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



def summary_figure(models, a):
    print('figure: summary metrics ...', flush=True)
    n = len(models)

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



def lejepa_figure(models, a, rng, c, ev0):
    print('figure: projections, spectra, patch PCA ...', flush=True)
    n = len(models)
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


if __name__ == '__main__':
    main()
