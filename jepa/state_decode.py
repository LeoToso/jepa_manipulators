#!/usr/bin/env python
"""Robot-state decoder: are the world model's predictions right in physical units?

    python -m jepa.state_decode --ckpt runs/msp_sigreg/model_final.pt --cache cache_128 --out eval/msp_sigreg/state_decoder

1. Train an MLP z -> [EE position (3), EE rotation (3, rotation vector relative to the mean
   orientation), gripper (1), joint angles (7)] on the frozen encoder's latents of the training
   episodes (targets standardized; the world model is not changed).
2. On test episodes, roll the predictor k = 1..K steps with the recorded actions and decode:
       Pred   decode(zhat_{t+k})          vs the measured state at t+k
       Recon  decode(encode(frame_{t+k})) vs the measured state  (the decoder's own limit)
       Copy   decode(z_t)                 vs the measured state at t+k ("the robot does not move")
   Errors in physical units: EE position [cm], EE rotation [deg, geodesic], gripper [0-2 range],
   joints [deg, mean over 7], overall and per task family.

Decoding the current state is easy when proprio is an encoder input; decoding the state k steps
ahead from a *predicted* latent is not: the predictor must have learned how the robot moves.

Outputs in --out: state_decoder.pt, state_decoder_metrics.json, and figures
    state_error_vs_k.png         error per horizon (Pred, Recon, Copy), all and per family
    state_rollout_<task>.png     predicted vs measured EE x / y / z / rotation for one test episode
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa.data import task_family
from jepa.evaluate import Evaluator
from jepa.models import torch_load

PARTS = [('ee_pos', 3), ('ee_rot', 3), ('gripper', 1), ('joints', 7)]


class StateDecoder(nn.Module):
    def __init__(self, latent_dim, out_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, z):
        return self.net(z)


def targets(ev, frames):
    return np.concatenate([ev.labels[k][frames].reshape(len(frames), -1) for k, _ in PARTS], 1).astype(np.float32)


def rotvec_angle_deg(a, b):
    """Geodesic angle between rotations given as rotation vectors, (N, 3) each -> (N,) degrees."""
    def quat(r):
        th = np.linalg.norm(r, axis=1, keepdims=True)
        axis = r / np.maximum(th, 1e-9)
        return np.concatenate([np.cos(th / 2), np.sin(th / 2) * axis], 1)
    dot = np.abs(np.sum(quat(a) * quat(b), 1)).clip(0, 1)
    return np.degrees(2 * np.arccos(dot))


def errors(pred, true):
    """Per-sample physical errors from de-standardized [pos, rot, grip, joints] vectors."""
    return {'ee_pos_cm': 100 * np.linalg.norm(pred[:, 0:3] - true[:, 0:3], axis=1),
            'ee_rot_deg': rotvec_angle_deg(pred[:, 3:6], true[:, 3:6]),
            'gripper': np.abs(pred[:, 6] - true[:, 6]),
            'joints_deg': np.degrees(np.abs(pred[:, 7:14] - true[:, 7:14]).mean(1))}


def train_state_decoder(ev, out, epochs, lr, max_frames, seed):
    c = ev.cache
    rng = np.random.RandomState(seed)
    tr = ev.episode_frames(c.split_episodes('train'))
    if len(tr) > max_frames:
        tr = np.sort(rng.choice(tr, max_frames, replace=False))
    va = ev.episode_frames(c.split_episodes('val'))
    print(f'encoding {len(tr)} train / {len(va)} val frames for the state decoder ...', flush=True)
    z_tr, z_va = ev.encode(tr), ev.encode(va)
    y_tr, y_va = targets(ev, tr), targets(ev, va)
    mean, std = y_tr.mean(0), np.maximum(y_tr.std(0), 1e-6)
    dev = ev.device
    Z, Y = torch.from_numpy(z_tr).to(dev), torch.from_numpy((y_tr - mean) / std).to(dev)
    Zv, Yv = torch.from_numpy(z_va).to(dev), torch.from_numpy((y_va - mean) / std).to(dev)
    dec = StateDecoder(Z.shape[1], Y.shape[1]).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = max(len(Z) // 512, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    best, best_sd = float('inf'), None
    for ep in range(1, epochs + 1):
        dec.train()
        perm = torch.randperm(len(Z), device=dev)
        for i in range(steps_per_epoch):
            j = perm[i * 512:(i + 1) * 512]
            loss = F.mse_loss(dec(Z[j]), Y[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        dec.eval()
        with torch.no_grad():
            vl = F.mse_loss(dec(Zv), Yv).item()
        if ep % 10 == 0 or ep == epochs:
            print(f'  state decoder epoch {ep:3d}/{epochs}  train={loss.item():.4f}  val={vl:.4f}', flush=True)
        if vl < best:
            best, best_sd = vl, {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()}
    dec.load_state_dict(best_sd)
    torch.save({'decoder': best_sd, 'latent_dim': Z.shape[1], 'out_dim': Y.shape[1],
                'mean': mean, 'std': std}, os.path.join(out, 'state_decoder.pt'))
    return dec.eval(), mean, std


@torch.no_grad()
def evaluate_state_decoder(ev, dec, mean, std, out, K, n_starts, seed, traj_steps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    c = ev.cache
    rng = np.random.RandomState(seed)
    D = lambda z: np.concatenate([dec(torch.from_numpy(z[i:i + 2048]).to(ev.device)).cpu().numpy()
                                  for i in range(0, len(z), 2048)]) * std + mean
    te = ev.episode_frames(c.split_episodes('test'))
    last = c.starts[c.episode_of_frame[te]] + c.lengths[c.episode_of_frame[te]] - 1
    starts = te[te + K * ev.fs <= last]
    starts = np.sort(rng.choice(starts, min(n_starts, len(starts)), replace=False))
    fam = np.array([task_family(c.tasks[t]) for t in ev.task_of_frame[starts]])
    z0 = ev.encode(starts)
    zhat = ev.rollout(z0, ev.action_chunks(starts, K))
    copy = D(z0)
    groups = {'all': np.ones(len(starts), bool)}
    groups.update({f: fam == f for f in sorted(set(fam))})
    qty = ['ee_pos_cm', 'ee_rot_deg', 'gripper', 'joints_deg']
    M = {'k': list(range(1, K + 1)), 'seconds_per_step': ev.fs / 10.0, 'n_starts': int(len(starts)),
         'error': {g: {src: {q: [] for q in qty} for src in ('pred', 'recon', 'copy')} for g in groups}}
    for k in range(1, K + 1):
        tgt = starts + k * ev.fs
        true = targets(ev, tgt)
        srcs = {'pred': D(zhat[:, k - 1]), 'recon': D(ev.encode(tgt)), 'copy': copy}
        for src, pred in srcs.items():
            e = errors(pred, true)
            for g, sel in groups.items():
                for q in qty:
                    M['error'][g][src][q].append(float(np.mean(e[q][sel])) if sel.any() else float('nan'))
    with open(os.path.join(out, 'state_decoder_metrics.json'), 'w') as fh:
        json.dump(M, fh, indent=1)

    name = os.path.basename(os.path.dirname(os.path.abspath(ev.ckpt_path)))
    units = {'ee_pos_cm': 'EE position error [cm]', 'ee_rot_deg': 'EE rotation error [deg]',
             'gripper': 'gripper error', 'joints_deg': 'joint error [deg]'}
    secs = np.array(M['k']) * M['seconds_per_step']
    fig, axes = plt.subplots(len(groups), 4, figsize=(17, 3.2 * len(groups)), squeeze=False)
    style = {'pred': ('C0', '-', 'prediction'), 'recon': ('C2', '--', 'reconstruction (decoder limit)'),
             'copy': ('C7', ':', '"robot does not move"')}
    for r, g in enumerate(groups):
        for cidx, q in enumerate(qty):
            ax = axes[r, cidx]
            for src, (col, ls, lab) in style.items():
                ax.plot(secs, M['error'][g][src][q], ls, color=col, marker='o', ms=3, label=lab)
            ax.set_title(f'{g}: {units[q]}', fontsize=9)
            ax.set_xlabel('prediction horizon [s]')
            ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f'{name}: decoded robot state of k-step predictions vs measured ({len(starts)} test starts)')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'state_error_vs_k.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # one test episode per task: predicted vs measured EE trajectory over an open-loop rollout
    for ti, task in enumerate(c.tasks):
        eps = [e for e in c.split_episodes('test') if c.episode_task[e] == ti]
        if not eps:
            continue
        e = eps[0]
        s0 = int(c.starts[e])
        n = min(traj_steps, (int(c.starts[e] + c.lengths[e]) - 1 - s0) // ev.fs)
        if n < 2:
            continue
        idx = s0 + np.arange(n + 1) * ev.fs
        zz = ev.encode(idx[:1])
        pred = np.concatenate([D(zz), D(ev.rollout(zz, ev.action_chunks(idx[:1], n))[0])])
        true, recon = targets(ev, idx), D(ev.encode(idx))
        t = np.arange(n + 1) * M['seconds_per_step']
        fig, axes = plt.subplots(1, 5, figsize=(18, 3.0))
        for j, (col, lab, scale) in enumerate(((0, 'EE x [cm]', 100), (1, 'EE y [cm]', 100), (2, 'EE z [cm]', 100),
                                               (5, 'EE rot z [deg]', 180 / np.pi), (6, 'gripper', 1))):
            axes[j].plot(t, true[:, col] * scale, 'k', lw=2, label='measured')
            axes[j].plot(t, recon[:, col] * scale, 'C2--', lw=1, label='decoded from encoding')
            axes[j].plot(t, pred[:, col] * scale, 'C0', lw=1.5, label='decoded from prediction')
            axes[j].set_title(lab, fontsize=9)
            axes[j].set_xlabel('time [s]')
            axes[j].grid(alpha=.3)
        axes[0].legend(fontsize=7)
        fig.suptitle(f'{name}: {task}, open-loop rollout of {n} steps from the episode start '
                     f'with the recorded actions', fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'state_rollout_{task}.png'), dpi=130, bbox_inches='tight')
        plt.close(fig)
    return M


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--max-train-frames', type=int, default=80000)
    p.add_argument('--rollout-steps', type=int, default=10)
    p.add_argument('--n-starts', type=int, default=1500)
    p.add_argument('--traj-steps', type=int, default=40, help='latent steps in the per-task trajectory plots')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--reuse', action='store_true', help='load <out>/state_decoder.pt instead of training')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ev = Evaluator(a.ckpt, a.cache, a.device)
    ev.ckpt_path = a.ckpt
    path = os.path.join(a.out, 'state_decoder.pt')
    if a.reuse and os.path.exists(path):
        ck = torch_load(path, ev.device)
        dec = StateDecoder(ck['latent_dim'], ck['out_dim']).to(ev.device)
        dec.load_state_dict(ck['decoder'])
        dec.eval()
        mean, std = ck['mean'], ck['std']
    else:
        dec, mean, std = train_state_decoder(ev, a.out, a.epochs, a.lr, a.max_train_frames, a.seed)
    M = evaluate_state_decoder(ev, dec, mean, std, a.out, a.rollout_steps, a.n_starts, a.seed, a.traj_steps)
    E = M['error']['all']
    for q in ('ee_pos_cm', 'ee_rot_deg', 'gripper'):
        print(f'{q:11s} pred / recon / copy at k=1: {E["pred"][q][0]:.2f} / {E["recon"][q][0]:.2f} / '
              f'{E["copy"][q][0]:.2f}   at k={a.rollout_steps}: {E["pred"][q][-1]:.2f} / '
              f'{E["recon"][q][-1]:.2f} / {E["copy"][q][-1]:.2f}')
    print(f'[done] {a.out}')


if __name__ == '__main__':
    main()
