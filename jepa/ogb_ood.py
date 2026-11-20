#!/usr/bin/env python
"""Out-of-distribution visual perturbations for the OGBench cube environments.

    MUJOCO_GL=egl python -m jepa.ogb_ood --out figures/ogb_ood     # preview: what the agent sees

Each perturbation changes the scene's appearance only (no physics), applied to the MuJoCo model
right after the environment is created, so the goal images rendered at reset show it too:
    floor        the table / floor (the image background) tinted wood-like orange
    floor_green  the table / floor tinted green
    light_dim    the lights at 40% intensity
    light_warm   the lights tinted warm (orange-yellow)
    cube         the cube blue instead of red
    camera       the agent camera moved by 3 cm (sideways and up)
    all          floor + light_warm + camera (cube colour unchanged)
Used by jepa.ogb_eval --ood NAME (and jepa.ogb_video --ood NAME).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

PERTURBATIONS = ['floor', 'floor_green', 'light_dim', 'light_warm', 'cube', 'camera', 'all']


def apply_ood(env, name):
    """Modify env's MuJoCo model in place. Returns a short description."""
    if not name or name == 'none':
        return 'nominal'
    names = ['floor', 'light_warm', 'camera'] if name == 'all' else [name]
    u = env.unwrapped
    if getattr(u, '_model', None) is None:
        env.reset(seed=0)
    m = u._model
    for n in names:
        if n == 'floor':
            mid = m.mat('grid').id
            m.mat_rgba[mid] = [0.85, 0.55, 0.30, 1.0]
        elif n == 'floor_green':
            mid = m.mat('grid').id
            m.mat_rgba[mid] = [0.45, 0.85, 0.50, 1.0]
        elif n == 'light_dim':
            m.light_diffuse[:] *= 0.4
            m.light_ambient[:] *= 0.4
        elif n == 'light_warm':
            m.light_diffuse[:] *= np.array([1.25, 0.95, 0.55])
            m.light_ambient[:] *= np.array([1.25, 0.95, 0.55])
        elif n == 'cube':
            u._cube_colors[0] = np.array([0.20, 0.45, 0.95, 1.0])
        elif n == 'camera':
            cid = m.cam('front_pixels').id
            m.cam_pos[cid] += np.array([0.0, 0.03, 0.03])
        else:
            raise SystemExit(f'unknown perturbation {n!r}; choose from {PERTURBATIONS}')
    return name


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env', default='visual-cube-single-v0')
    p.add_argument('--out', required=True)
    p.add_argument('--task', type=int, default=1)
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from jepa.ogb_data import _env
    os.makedirs(a.out, exist_ok=True)
    rows = ['none'] + PERTURBATIONS
    fig, axes = plt.subplots(2, len(rows), figsize=(1.9 * len(rows), 4.0), squeeze=False)
    for j, n in enumerate(rows):
        env = _env(a.env)
        apply_ood(env, n)
        ob, info = env.reset(seed=1, options=dict(task_id=a.task))
        for r, (im, lab) in enumerate(((ob, 'start'), (info['goal'], 'goal'))):
            ax = axes[r, j]
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title('nominal' if n == 'none' else n, fontsize=9)
            if j == 0:
                ax.set_ylabel(lab)
        env.close()
    fig.suptitle('Visual perturbations (what the agent sees, 64 x 64)', fontsize=10)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(a.out, f'fig_ood_preview.{ext}'), dpi=200, bbox_inches='tight')
    print(f'[done] {os.path.join(a.out, "fig_ood_preview.png")}')


if __name__ == '__main__':
    main()
