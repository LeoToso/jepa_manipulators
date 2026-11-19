#!/usr/bin/env python
"""The latent-space analyses of the Franka models, for the OGBench cube world models.

    python -m jepa.ogb_analysis --cache /mnt/t7shield/cache_ogb_cube_single --out figures/ogb_latent \\
        --model "MSP+SIG=runs/ogb/msp_sigreg" --model "MSP+GCB+SIG=runs/ogb/msp_gcb_sigreg" \\
        --model "MSP+SIG+SF=runs/ogb/msp_sigreg_sf"

Physical quantities: effector position, cube position (both 3-D, cm) and gripper opening (0-1).
Latents are fitted on training episodes and scored on the validation + test episodes.
Writes (PDF + PNG, numbers in ogb_latent.json):
  fig_ogb_pc_error   error of a linear readout from the k leading principal directions (fit on
                     training episodes), effector / cube / gripper: how much of the task state the
                     main directions of the latent hold (the Franka fig_latent_pc_error)
  fig_ogb_decoded    an MLP decoder on frozen latents: decoded vs measured effector height, cube
                     lateral position and gripper opening on held-out frames (fig_latent_decoded)
  fig_ogb_prediction open-loop predictions with the recorded actions: decoded effector and cube
                     position error after k latent steps, against the decoder limit (decoding the
                     true future latent) and "nothing moves" (decoding the current latent)
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
import torch.nn as nn  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter  # noqa: E402

from jepa.data import Cache  # noqa: E402
from jepa.models import load_checkpoint  # noqa: E402

MODEL_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False, 'figure.facecolor': 'white',
    'savefig.facecolor': 'white', 'lines.linewidth': 2.0, 'pdf.fonttype': 42,
})
# target columns: effector xyz (m), cube xyz (m), gripper opening
QUANT = (('effector position', slice(0, 3), 'cm', 100.0), ('cube position', slice(3, 6), 'cm', 100.0),
         ('gripper opening', slice(6, 7), '% of range', 100.0))


def save(fig, out, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {name}.pdf/.png', flush=True)


def frames_of(c, eps, stride):
    return np.concatenate([np.arange(c.starts[e], c.starts[e] + c.lengths[e], stride) for e in eps])


class Model:
    def __init__(self, label, run, c, device):
        self.label = label
        self.m, ck = load_checkpoint(run if run.endswith('.pt') else os.path.join(run, 'model_final.pt'), device)
        dc = ck['config']['data']
        self.fs, self.dev, self.c = int(dc['frame_skip']), device, c
        st = ck['action_stats']
        self.act = ((c.actions(dc['action_mode']) - np.array(st['mean'], np.float32))
                    / np.array(st['std'], np.float32)).astype(np.float32)
        ps = ck.get('proprio_stats')
        self.prop = None
        if ps is not None:
            from jepa.data import proprio_array
            self.prop = ((proprio_array(c, ps['keys']) - np.array(ps['mean'], np.float32))
                         / np.array(ps['std'], np.float32)).astype(np.float32)
        self.ep_start = c.starts[c.episode_of_frame]

    @torch.no_grad()
    def encode(self, frames, bs=1024):
        out = []
        im = self.c.images
        for i in range(0, len(frames), bs):
            f = frames[i:i + bs]
            prev = np.maximum(f - self.fs, self.ep_start[f])
            o = torch.from_numpy(np.ascontiguousarray(im[f])).permute(0, 3, 1, 2).to(self.dev)
            p = torch.from_numpy(np.ascontiguousarray(im[prev])).permute(0, 3, 1, 2).to(self.dev)
            pr = None if self.prop is None else torch.from_numpy(self.prop[f]).to(self.dev)
            out.append(self.m.encode(o, p, pr).float().cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def rollout(self, z0, starts, K, bs=1024):
        """(n, d) latents at frames `starts` -> (n, K, d) open-loop predictions with recorded actions."""
        rows = starts[:, None] + np.arange(K * self.fs)[None]
        a = self.act[rows].reshape(len(starts), K, -1)
        out = []
        for i in range(0, len(starts), bs):
            out.append(self.m.rollout(torch.from_numpy(z0[i:i + bs]).to(self.dev),
                                      torch.from_numpy(a[i:i + bs]).to(self.dev)).float().cpu())
        return torch.cat(out).numpy()


def fit_decoder(z, y, device, steps=3000, seed=0):
    torch.manual_seed(seed)
    mu, sd = y.mean(0), y.std(0) + 1e-6
    zm, zs = z.mean(0), z.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(z.shape[1], 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, 512),
                        nn.LayerNorm(512), nn.GELU(), nn.Linear(512, y.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Z = torch.from_numpy((z - zm) / zs).float().to(device)
    Y = torch.from_numpy((y - mu) / sd).float().to(device)
    for _ in range(steps):
        idx = torch.randint(0, len(Z), (1024,), device=device)
        loss = ((net(Z[idx]) - Y[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    net.eval()

    @torch.no_grad()
    def dec(zz):
        out = [net(torch.from_numpy((zz[i:i + 8192] - zm) / zs).float().to(device)).cpu().numpy()
               for i in range(0, len(zz), 8192)]
        return np.concatenate(out) * sd + mu
    return dec


def err(pred, true, sl, scale):
    d = (pred[:, sl] - true[:, sl]) * scale
    return float(np.sqrt((d ** 2).sum(1).mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model', action='append', required=True, help='LABEL=RUN_DIR')
    p.add_argument('--train-episodes', type=int, default=300, help='training episodes used for PCA / fits')
    p.add_argument('--stride', type=int, default=4, help='frame stride (training and held-out frames)')
    p.add_argument('--horizon', type=int, default=20, help='prediction horizon in latent steps')
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device if (a.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    rng = np.random.RandomState(a.seed)
    c = Cache(a.cache)
    low = c.low
    Y_all = np.concatenate([low['x_ee'][:, :3], low['obj_pos'][:, :3], low['gripper'][:, :1]], 1).astype(np.float32)
    tr_eps = sorted(rng.choice(c.split_episodes('train'), min(a.train_episodes, len(c.split_episodes('train'))),
                               replace=False))
    ho_eps = c.split_episodes('val') + c.split_episodes('test')
    tr, ho = frames_of(c, tr_eps, a.stride), frames_of(c, ho_eps, a.stride)
    print(f'{len(tr)} training frames ({len(tr_eps)} episodes), {len(ho)} held-out frames ({len(ho_eps)} episodes)')
    Ytr, Yho = Y_all[tr], Y_all[ho]
    models = [Model(*spec.split('=', 1), c, dev) for spec in a.model]
    record = {}
    ks = np.array([1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100, 120, 140, 160, 192, 256, 384, 512])

    # ---- encode, PCA error curves, decoders ------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for j, m in enumerate(models):
        print(f'{m.label}: encoding ...', flush=True)
        m.z_tr, m.z_ho = m.encode(tr), m.encode(ho)
        mu = m.z_tr.mean(0)
        vt = np.linalg.svd(m.z_tr - mu, full_matrices=False)[2]
        kk = ks[ks <= vt.shape[0]]
        curves = {q: [] for q, *_ in QUANT}
        for k in kk:
            Xtr = np.c_[(m.z_tr - mu) @ vt[:k].T, np.ones(len(tr))]
            Xho = np.c_[(m.z_ho - mu) @ vt[:k].T, np.ones(len(ho))]
            pred = Xho @ np.linalg.lstsq(Xtr, Ytr, rcond=None)[0]
            for q, sl, _, sc in QUANT:
                curves[q].append(err(pred, Yho, sl, sc))
        for ax, (q, sl, unit, sc) in zip(axes, QUANT):
            ax.plot(kk, curves[q], color=MODEL_COLORS[j % 3], marker='o', ms=2.5, label=m.label)
        record.setdefault(m.label, {})['pc_error'] = {'k': kk.tolist(), **curves}
        print(f'{m.label}: fitting the state decoder ...', flush=True)
        m.dec = fit_decoder(m.z_tr, Ytr, dev, seed=a.seed)
        pred = m.dec(m.z_ho)
        record[m.label]['decoder_rmse'] = {q: err(pred, Yho, sl, sc) for q, sl, _, sc in QUANT}
        print('   decoder RMSE on held-out frames: ' + ', '.join(
            f'{q} {v:.2f}' for q, v in record[m.label]['decoder_rmse'].items()), flush=True)
    for ax, (q, sl, unit, sc) in zip(axes, QUANT):
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(1, kk.max())
        ax.set_xticks([1] + list(range(25, int(kk.max()) + 1, 25)) + ([int(kk.max())] if kk.max() % 25 else []))
        ax.set_xlabel('number of leading principal directions')
        ax.set_ylabel(f'{q} error [{unit}]')
        ax.set_title(q)
    h, l_ = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(h, l_, loc='lower center', ncol=len(models), fontsize=9)
    save(fig, a.out, 'fig_ogb_pc_error')

    # ---- decoded vs measured -----------------------------------------------------------------------
    pts = rng.choice(len(ho), min(3000, len(ho)), replace=False)
    rows = (('effector height', 2, 100.0, 'cm'), ('cube y position', 4, 100.0, 'cm'),
            ('gripper opening', 6, 1.0, '0 open - 1 closed'))
    fig, axes = plt.subplots(len(rows), len(models), figsize=(3.4 * len(models), 3.3 * len(rows)), squeeze=False)
    for j, m in enumerate(models):
        pred = m.dec(m.z_ho)
        for r, (name, col, sc, unit) in enumerate(rows):
            ax = axes[r, j]
            yt, yp = Yho[pts, col] * sc, pred[pts, col] * sc
            lo, hi = np.percentile(np.r_[yt, yp], [0.5, 99.5])
            ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls='--')
            ax.scatter(yt, yp, s=4, color=MODEL_COLORS[j % 3], alpha=0.35, linewidths=0)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_title(f'{m.label}: {name} decoded from z', fontsize=9)
            ax.set_xlabel(f'measured {name} [{unit}]')
            if j == 0:
                ax.set_ylabel(f'decoded {name} [{unit}]')
    fig.tight_layout(h_pad=2.0)
    save(fig, a.out, 'fig_ogb_decoded')

    # ---- open-loop prediction -----------------------------------------------------------------------
    K = a.horizon
    fs = models[0].fs
    ep_end = c.starts[c.episode_of_frame] + c.lengths[c.episode_of_frame] - 1
    starts = ho[ho + K * fs <= ep_end[ho]]
    starts = starts[rng.choice(len(starts), min(3000, len(starts)), replace=False)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ksteps = np.arange(1, K + 1)
    for j, m in enumerate(models):
        z0 = m.encode(starts)
        zh = m.rollout(z0, starts, K)
        res = {}
        for ax, (q, sl, unit, sc) in zip(axes, QUANT[:2]):
            pe, re_, ce = [], [], []
            copy = m.dec(z0)
            for k in ksteps:
                tf = starts + k * fs
                pe.append(err(m.dec(zh[:, k - 1]), Y_all[tf], sl, sc))
                re_.append(err(m.dec(m.encode(tf)), Y_all[tf], sl, sc))
                ce.append(err(copy, Y_all[tf], sl, sc))
            col = MODEL_COLORS[j % 3]
            ax.plot(ksteps, pe, color=col, lw=2.0, label=f'{m.label}: prediction')
            ax.plot(ksteps, re_, color=col, lw=1.0, ls=':', label=f'{m.label}: decoder limit')
            if j == 0:
                ax.plot(ksteps, ce, color=INK2, lw=1.2, ls='--', label='nothing moves')
            res[q] = {'pred': pe, 'decoder_limit': re_, 'copy': ce}
        record[m.label]['prediction'] = {'k_latent_steps': ksteps.tolist(), 'env_steps_per_latent_step': fs, **res}
    for ax, (q, sl, unit, sc) in zip(axes, QUANT[:2]):
        ax.set_xlabel(f'prediction horizon [latent steps of {fs} env steps]')
        ax.set_ylabel(f'{q} error [{unit}]')
        ax.set_title(f'{q}: open-loop prediction with recorded actions')
        ax.set_ylim(bottom=0)
    h, l_ = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.legend(h, l_, loc='lower center', ncol=3, fontsize=8)
    save(fig, a.out, 'fig_ogb_prediction')
    with open(os.path.join(a.out, 'ogb_latent.json'), 'w') as fh:
        json.dump(record, fh, indent=1)
    print(f'[done] {a.out}')


if __name__ == '__main__':
    main()
