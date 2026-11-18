#!/usr/bin/env python
"""Look at an OGBench cube cache before training.

    MUJOCO_GL=egl python -m jepa.ogb_viz --cache /mnt/t7shield/cache_ogb_cube_single --out figures/ogb_data

Writes into --out:
  frames.png      rows = episodes, columns = frames every --every steps (what the models see, 64 x 64)
  episode.gif     one episode at 4x size, with its step and cube height
  stats.png       action histograms (5 dims), cube and effector coverage of the table (x-y), cube
                  height distribution (lifts), gripper opening, episode lengths
  timeline.png    one episode over time: effector and cube position, gripper opening vs command
  eval_tasks.png  OGBench's 5 evaluation tasks: start image and goal image (needs the environment)
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jepa.data import Cache  # noqa: E402

INK, INK2, BLUE, ORANGE, AQUA = '#0b0b0b', '#52514e', '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
ACTION_NAMES = ['dx', 'dy', 'dz', 'dyaw', 'dgripper']


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {name}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--episodes', type=int, default=6, help='rows of frames.png')
    p.add_argument('--every', type=int, default=50, help='steps between columns of frames.png')
    p.add_argument('--episode', type=int, default=0, help='episode shown in episode.gif / timeline.png')
    p.add_argument('--gif-steps', type=int, default=300)
    p.add_argument('--no-env', action='store_true', help='skip eval_tasks.png (no MuJoCo rendering)')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c = Cache(a.cache)
    low, rng = c.low, np.random.RandomState(a.seed)
    acts = c.actions('stored')
    n_ep = len(c.lengths)
    print(f'{n_ep} episodes, {len(acts)} frames, images {c.images.shape[1:]}; splits '
          + ', '.join(f'{k}={len(v)}' for k, v in c.meta['splits'].items()))

    # frames.png
    eps = np.sort(rng.choice(n_ep, min(a.episodes, n_ep), replace=False))
    cols = int(min(c.lengths[eps].min(), 1001) // a.every) + 1
    fig, axes = plt.subplots(len(eps), cols, figsize=(1.1 * cols, 1.2 * len(eps)), squeeze=False)
    for r, e in enumerate(eps):
        for k in range(cols):
            f = int(c.starts[e]) + min(k * a.every, int(c.lengths[e]) - 1)
            ax = axes[r, k]
            ax.imshow(c.images[f])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f't={k * a.every}', fontsize=7)
            if k == 0:
                ax.set_ylabel(f'ep {e}', fontsize=7)
    fig.suptitle('Frames the models see (64 x 64), every %d steps' % a.every, fontsize=10)
    save(fig, a.out, 'frames.png')

    # episode.gif
    from PIL import Image, ImageDraw
    e = a.episode
    s0, L = int(c.starts[e]), int(c.lengths[e])
    imgs = []
    for t in range(0, min(L, a.gif_steps)):
        im = Image.fromarray(np.asarray(c.images[s0 + t])).resize((256, 256), Image.NEAREST)
        d = ImageDraw.Draw(im)
        d.text((4, 4), f't={t}  cube z={100 * low["obj_pos"][s0 + t, 2]:.1f} cm  grip={low["gripper"][s0 + t, 0]:.2f}',
               fill=(255, 255, 255))
        imgs.append(im)
    imgs[0].save(os.path.join(a.out, 'episode.gif'), save_all=True, append_images=imgs[1:], duration=60, loop=0)
    print('  wrote episode.gif')

    # stats.png
    sub = rng.choice(len(acts), min(200000, len(acts)), replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    for k in range(5):
        ax = axes[0, k]
        ax.hist(acts[sub, k], bins=60, color=BLUE, alpha=0.85)
        ax.set_title(f'action {ACTION_NAMES[k]}')
        ax.set_yscale('log')
    ax = axes[1, 0]
    obj = low['obj_pos']
    ax.hist2d(obj[sub, 0], obj[sub, 1], bins=60, cmap='Blues')
    ax.set_title('cube x-y coverage')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax = axes[1, 1]
    ee = low['x_ee']
    ax.hist2d(ee[sub, 0], ee[sub, 1], bins=60, cmap='Oranges')
    ax.set_title('effector x-y coverage')
    ax.set_xlabel('x [m]')
    ax = axes[1, 2]
    ax.hist(100 * obj[sub, 2], bins=80, color=AQUA)
    lifted = (obj[:, 2] > 0.04).mean()
    ax.set_yscale('log')
    ax.set_title(f'cube height (lifted > 4 cm: {100 * lifted:.1f}% of frames)')
    ax.set_xlabel('z [cm]')
    ax = axes[1, 3]
    ax.hist(low['gripper'][sub, 0], bins=50, color=INK2)
    ax.set_title('gripper opening (0 open, 1 closed)')
    ax = axes[1, 4]
    ax.hist(c.lengths, bins=30, color=INK2)
    ax.set_title('episode length [steps]')
    fig.tight_layout()
    save(fig, a.out, 'stats.png')

    # timeline.png
    t = np.arange(L)
    sl = slice(s0, s0 + L)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for k, nm, col in ((0, 'x', BLUE), (1, 'y', ORANGE), (2, 'z', AQUA)):
        axes[0].plot(t, 100 * ee[sl, k], color=col, label=f'effector {nm}')
        axes[1].plot(t, 100 * obj[sl, k], color=col, label=f'cube {nm}')
    axes[0].set_ylabel('effector [cm]')
    axes[1].set_ylabel('cube [cm]')
    axes[2].plot(t, low['gripper'][sl, 0], color=INK, label='gripper opening')
    axes[2].plot(t, acts[sl, 4], color=ORANGE, lw=0.8, alpha=0.8, label='action dgripper')
    axes[2].set_ylabel('gripper')
    axes[2].set_xlabel('step')
    for ax in axes:
        ax.legend(fontsize=7, loc='upper right', ncol=3)
    fig.suptitle(f'Episode {e} over time', fontsize=10)
    fig.tight_layout()
    save(fig, a.out, 'timeline.png')

    # eval_tasks.png
    if not a.no_env:
        from jepa.ogb_data import _env
        env = _env(c.meta.get('env_name', 'visual-cube-single-v0'))
        fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
        for k in range(5):
            ob, info = env.reset(seed=a.seed, options=dict(task_id=k + 1))
            for r, (im, nm) in enumerate(((ob, 'start'), (info['goal'], 'goal'))):
                ax = axes[r, k]
                ax.imshow(im)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(f'task {k + 1}', fontsize=9)
                if k == 0:
                    ax.set_ylabel(nm)
        fig.suptitle('OGBench evaluation tasks: start and goal image', fontsize=10)
        fig.tight_layout()
        save(fig, a.out, 'eval_tasks.png')
    print(f'[done] {a.out}')


if __name__ == '__main__':
    main()
