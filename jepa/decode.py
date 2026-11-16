#!/usr/bin/env python
"""Pixel decoder for a trained world model: what do its (predicted) latents look like?

    python -m jepa.decode --ckpt runs/msp_sigreg/model_final.pt --cache cache_128 --out eval/msp_sigreg/decoder

1. Train a convolutional decoder z -> image on the frozen encoder's latents of the training
   episodes (architecture as JEPA_control's experiments/train_pixel_decoder.py, one more
   upsampling stage for 128 px). The world model is not changed.
2. On test episodes, roll the predictor k = 1..K steps with the recorded actions and decode:
       GT     the true frame at t+k
       Recon  decode(encode(GT))   -- the decoder's own limit (best a prediction can look)
       Pred   decode(zhat_{t+k})   -- the world model's prediction
       Copy   decode(z_t)          -- "nothing changes" baseline
   and report PSNR / SSIM per k for Pred vs GT, Recon vs GT, Copy vs GT and Pred vs Recon
   (prediction error with the decoder's limits removed), over the whole image and over the
   pixels that move between t and t+k (gripper, object), where the static background cannot
   hide errors and "nothing changes" is wrong by construction.

Outputs in --out: decoder.pt, decoder_metrics.json, and figures
    decoder_error_vs_k.png       PSNR / SSIM per horizon (Pred, Recon, Copy)
    rollout_<task>.png           GT / Recon / Pred strips over k, one test episode per task
    rollout_<task>.gif           the same as an animation, side by side
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa.evaluate import Evaluator


class ConvDecoder(nn.Module):
    """z -> (3, S, S) in [0, 1]: linear to 4x4, then ConvTranspose/BN/ReLU stages (JEPA_control)."""

    def __init__(self, latent_dim, image_size=128, base_ch=256):
        super().__init__()
        n_up = int(np.log2(image_size // 4))
        if 4 * 2 ** n_up != image_size:
            raise ValueError('image_size must be 4 * 2^n')
        self.base_ch = base_ch
        self.fc = nn.Linear(latent_dim, base_ch * 4 * 4)
        layers, ch = [], base_ch
        for i in range(n_up):
            last = i == n_up - 1
            out = 3 if last else max(ch // 2, 32)
            layers.append(nn.ConvTranspose2d(ch, out, 4, 2, 1))
            if not last:
                layers += [nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
            ch = out
        layers.append(nn.Sigmoid())
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        return self.deconv(self.fc(z).view(z.shape[0], self.base_ch, 4, 4))


def psnr(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


def ssim(a, b, window=7):
    """Mean SSIM per image (uniform window), images in [0, 1], shape (N, 3, H, W)."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    pool = lambda x: F.avg_pool2d(x, window, 1, window // 2, count_include_pad=False)
    mu_a, mu_b = pool(a), pool(b)
    va = pool(a * a) - mu_a ** 2
    vb = pool(b * b) - mu_b ** 2
    cov = pool(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return s.flatten(1).mean(1)


def moving_mask(gt_now, gt_later, thresh=0.06, dilate=5):
    """Pixels that change between t and t+k (gripper, object), dilated: (N, 1, H, W) in {0, 1}."""
    m = ((gt_later - gt_now).abs().mean(1, keepdim=True) > thresh).float()
    return F.max_pool2d(m, dilate, 1, dilate // 2)


def masked_psnr(a, b, mask):
    """Per-image PSNR over the masked pixels (callers drop images with an empty mask)."""
    w = mask.expand_as(a)
    mse = (((a - b) ** 2) * w).flatten(1).sum(1) / w.flatten(1).sum(1).clamp_min(1.0)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


MIN_MOVING_PIXELS = 30  # images where fewer pixels move are left out of the moving-pixel PSNR


def frames_tensor(cache, idx):
    return torch.from_numpy(np.ascontiguousarray(cache.images[idx])).permute(0, 3, 1, 2).float() / 255.


def train_decoder(ev, out, epochs, batch_size, lr, max_frames, seed):
    c = ev.cache
    rng = np.random.RandomState(seed)
    tr = ev.episode_frames(c.split_episodes('train'))
    if len(tr) > max_frames:
        tr = np.sort(rng.choice(tr, max_frames, replace=False))
    va = ev.episode_frames(c.split_episodes('val'), stride=2)
    print(f'encoding {len(tr)} train / {len(va)} val frames for the decoder ...', flush=True)
    z_tr, z_va = torch.from_numpy(ev.encode(tr)), torch.from_numpy(ev.encode(va))
    dec = ConvDecoder(z_tr.shape[1], int(c.meta['image_size'])).to(ev.device)
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    steps = epochs * max(len(tr) // batch_size, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    best, best_sd, hist = float('inf'), None, []
    for ep in range(1, epochs + 1):
        dec.train()
        perm = rng.permutation(len(tr))
        tot = 0.0
        for i in range(0, len(perm) - batch_size + 1, batch_size):
            j = np.sort(perm[i:i + batch_size])
            x = frames_tensor(c, tr[j]).to(ev.device)
            loss = F.mse_loss(dec(z_tr[j].to(ev.device)), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(j)
        dec.eval()
        with torch.no_grad():
            vl = sum(F.mse_loss(dec(z_va[i:i + 512].to(ev.device)),
                                frames_tensor(c, va[i:i + 512]).to(ev.device), reduction='sum').item()
                     for i in range(0, len(va), 512)) / (len(va) * 3 * c.meta['image_size'] ** 2)
        hist.append({'epoch': ep, 'train_mse': tot / len(perm), 'val_mse': vl})
        print(f'  decoder epoch {ep:3d}/{epochs}  train={tot / len(perm):.5f}  val={vl:.5f}', flush=True)
        if vl < best:
            best, best_sd = vl, {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()}
    dec.load_state_dict(best_sd)
    torch.save({'decoder': best_sd, 'latent_dim': z_tr.shape[1], 'image_size': c.meta['image_size'],
                'history': hist}, os.path.join(out, 'decoder.pt'))
    return dec.eval()


@torch.no_grad()
def evaluate_decoder(ev, dec, out, K, n_starts, seed, gif_steps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    c = ev.cache
    rng = np.random.RandomState(seed)
    te = ev.episode_frames(c.split_episodes('test'))
    last = c.starts[c.episode_of_frame[te]] + c.lengths[c.episode_of_frame[te]] - 1
    starts = te[te + K * ev.fs <= last]
    starts = np.sort(rng.choice(starts, min(n_starts, len(starts)), replace=False))
    z0 = ev.encode(starts)
    zhat = ev.rollout(z0, ev.action_chunks(starts, K))                     # (n, K, d)
    D = lambda z: torch.cat([dec(torch.from_numpy(z[i:i + 256]).to(ev.device)).cpu()
                             for i in range(0, len(z), 256)])
    copy_img = D(z0)
    keys = ['pred_vs_gt', 'recon_vs_gt', 'copy_vs_gt', 'pred_vs_recon']
    M = {'k': list(range(1, K + 1)), 'psnr': {k: [] for k in keys}, 'ssim': {k: [] for k in keys},
         'psnr_moving': {k: [] for k in keys}, 'moving_fraction': [], 'moving_images': []}
    gt0 = frames_tensor(c, starts)
    for k in range(1, K + 1):
        tgt = starts + k * ev.fs
        gt = frames_tensor(c, tgt)
        mask = moving_mask(gt0, gt)
        M['moving_fraction'].append(float(mask.mean()))
        # an empty mask (robot and object still) would score as a perfect 100 dB: leave it out
        valid = mask.flatten(1).sum(1) >= MIN_MOVING_PIXELS
        M['moving_images'].append(int(valid.sum()))
        recon = D(ev.encode(tgt))
        pred = D(zhat[:, k - 1])
        for key, (a, b) in zip(keys, ((pred, gt), (recon, gt), (copy_img, gt), (pred, recon))):
            M['psnr'][key].append(float(psnr(a, b).mean()))
            M['ssim'][key].append(float(ssim(a, b).mean()))
            M['psnr_moving'][key].append(float(masked_psnr(a[valid], b[valid], mask[valid]).mean())
                                         if valid.any() else float('nan'))
    M['n_starts'], M['frame_skip'] = int(len(starts)), ev.fs
    with open(os.path.join(out, 'decoder_metrics.json'), 'w') as fh:
        json.dump(M, fh, indent=1)

    name = os.path.basename(os.path.dirname(os.path.abspath(ev.ckpt_path))) if hasattr(ev, 'ckpt_path') else ''
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.6))
    styles = {'pred_vs_gt': ('C0', '-', 'prediction vs GT'), 'recon_vs_gt': ('C2', '--', 'reconstruction vs GT (decoder limit)'),
              'copy_vs_gt': ('C7', ':', '"nothing changes" vs GT'), 'pred_vs_recon': ('C3', '-.', 'prediction vs reconstruction')}
    titles = {'psnr': 'PSNR, whole image', 'ssim': 'SSIM, whole image',
              'psnr_moving': 'PSNR on pixels that move between t and t+k'}
    for ax, metric in zip(axes, ('psnr', 'ssim', 'psnr_moving')):
        for key, (col, ls, lab) in styles.items():
            ax.plot(M['k'], M[metric][key], ls, color=col, marker='o', ms=3, label=lab)
        ax.set_xlabel(f'prediction horizon k (latent steps of {ev.fs} frames)')
        ax.set_ylabel(metric.upper().replace('_', ' '))
        ax.set_title(titles[metric], fontsize=9)
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f'{name}: decoded predictions vs ground truth ({len(starts)} test starts)')
    fig.savefig(os.path.join(out, 'decoder_error_vs_k.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # one test episode per task: strips and a GIF
    show = list(range(0, K + 1, max(1, K // 5)))
    for ti, task in enumerate(c.tasks):
        eps = [e for e in c.split_episodes('test') if c.episode_task[e] == ti]
        if not eps:
            continue
        e = eps[0]
        s0 = int(c.starts[e] + c.lengths[e] // 4)
        n = min(gif_steps, (int(c.starts[e] + c.lengths[e]) - 1 - s0) // ev.fs)
        if n < 1:
            continue
        zz = ev.encode(np.array([s0]))
        zh = ev.rollout(zz, ev.action_chunks(np.array([s0]), n))[0]
        idx = s0 + np.arange(n + 1) * ev.fs
        gt = frames_tensor(c, idx)
        recon = D(ev.encode(idx))
        pred = torch.cat([D(zz), D(zh)])
        cols = [k for k in show if k <= n]
        fig, axes = plt.subplots(3, len(cols), figsize=(1.9 * len(cols), 6.0))
        for j, k in enumerate(cols):
            for r, (img, lab) in enumerate(((gt, 'GT'), (recon, 'Recon'), (pred, 'Pred'))):
                ax = axes[r, j]
                ax.imshow(img[k].permute(1, 2, 0).numpy())
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(lab)
                if r == 0:
                    ax.set_title(f'k={k} ({k * ev.fs / 10:.1f}s)', fontsize=8)
        fig.suptitle(f'{name}: {task} (open-loop rollout with the recorded actions)', fontsize=9)
        fig.savefig(os.path.join(out, f'rollout_{task}.png'), dpi=130, bbox_inches='tight')
        plt.close(fig)
        frames = []
        for k in range(n + 1):
            row = torch.cat([gt[k], recon[k], pred[k]], dim=2).permute(1, 2, 0).numpy()
            frames.append(Image.fromarray((row * 255).astype(np.uint8)).resize((row.shape[1] * 2, row.shape[0] * 2)))
        frames[0].save(os.path.join(out, f'rollout_{task}.gif'), save_all=True, append_images=frames[1:],
                       duration=int(100 * ev.fs * 2), loop=0)
    return M


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--max-train-frames', type=int, default=60000)
    p.add_argument('--rollout-steps', type=int, default=10)
    p.add_argument('--n-starts', type=int, default=500)
    p.add_argument('--gif-steps', type=int, default=20, help='latent steps in the per-task rollout GIFs')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--reuse', action='store_true', help='load <out>/decoder.pt instead of training')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ev = Evaluator(a.ckpt, a.cache, a.device)
    ev.ckpt_path = a.ckpt
    path = os.path.join(a.out, 'decoder.pt')
    if a.reuse and os.path.exists(path):
        ck = torch.load(path, map_location=ev.device)
        dec = ConvDecoder(ck['latent_dim'], ck['image_size']).to(ev.device)
        dec.load_state_dict(ck['decoder'])
        dec.eval()
    else:
        dec = train_decoder(ev, a.out, a.epochs, a.batch_size, a.lr, a.max_train_frames, a.seed)
    M = evaluate_decoder(ev, dec, a.out, a.rollout_steps, a.n_starts, a.seed, a.gif_steps)
    for metric in ('psnr', 'psnr_moving'):
        print(f'{metric} pred / recon / copy at k=1:', [round(M[metric][k][0], 2) for k in ('pred_vs_gt', 'recon_vs_gt', 'copy_vs_gt')],
              ' at k=K:', [round(M[metric][k][-1], 2) for k in ('pred_vs_gt', 'recon_vs_gt', 'copy_vs_gt')])
    print(f'[done] {a.out}')


if __name__ == '__main__':
    main()
