#!/usr/bin/env python
"""Disturbances during execution for the OGBench cube evaluation (same appearance, off-distribution
states). Used by jepa.ogb_eval --disturb SPEC; preview with

    python -m jepa.ogb_disturb --dp runs/ogb/dp --out figures/ogb_disturb --task 2

SPEC (strength after the colon, optional):
    none            no disturbance
    push[:d]        once per episode the cube is knocked away from the goal by d metres (default
                    0.08) and dropped on the table: the first time it is lifted above 4 cm (i.e. out
                    of the gripper during transport), or at step 80 if it has not been lifted by then
    noise[:s]       Gaussian noise of std s (default 0.2, in the [-1, 1] action range) on the motion
                    dims of every executed action (dx, dy, dz, dyaw; not the gripper)
    burst[:p]       at every step, with probability p (default 0.03), a burst of 5 uniformly random
                    actions replaces the planned ones (a bump / controller glitch)
The random numbers depend only on the episode seed, so every method meets the same disturbances.
"""
from __future__ import annotations

import argparse
import os

import numpy as np


class Disturbance:
    def __init__(self, spec, seed):
        name, _, val = (spec or 'none').partition(':')
        self.name = name
        self.strength = float(val) if val else {'push': 0.08, 'noise': 0.2, 'burst': 0.03}.get(name, 0.0)
        if name not in ('none', 'push', 'noise', 'burst'):
            raise SystemExit(f'unknown disturbance {spec!r}: none, push[:d], noise[:s], burst[:p]')
        self.rng = np.random.RandomState(seed + 7919)
        self.pushed, self.burst_left, self.events = False, 0, []

    def action(self, a, t):
        """The action actually executed at step t (planned action a)."""
        a = np.asarray(a, np.float64).copy()
        if self.name == 'noise':
            a[:4] += self.rng.normal(0.0, self.strength, 4)
        elif self.name == 'burst':
            start = self.rng.rand() < self.strength          # always drawn: same sequence for every method
            rand_a = self.rng.uniform(-1, 1, a.shape)
            if self.burst_left == 0 and start:
                self.burst_left = 5
                self.events.append(t)
            if self.burst_left > 0:
                a = rand_a
                self.burst_left -= 1
        return np.clip(a, -1, 1)

    def after_step(self, env, t):
        """Called after each env step; may change the state (push). Returns True if it did."""
        if self.name != 'push' or self.pushed:
            return False
        import mujoco
        u = env.unwrapped
        j = u._data.joint('object_joint_0')
        pos = j.qpos[:3].copy()
        if pos[2] <= 0.04 and t < 80:
            return False
        goal = np.asarray(u.cur_task_info['goal_xyzs'][0][:2])
        away = pos[:2] - goal
        away = away / (np.linalg.norm(away) + 1e-9)
        ang = self.rng.uniform(-np.pi / 3, np.pi / 3)                     # within 60 deg of "away from goal"
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        lo, hi = u._object_sampling_bounds
        new_xy = np.clip(pos[:2] + self.strength * rot @ away, lo, hi)
        j.qpos[:3] = [new_xy[0], new_xy[1], 0.02]
        j.qvel[:] = 0.0
        mujoco.mj_forward(u._model, u._data)
        self.pushed = True
        self.events.append(t)
        return True


def main():
    """Preview: the policy alone on one task under each disturbance, as a frame strip per condition
    (the frames right after a disturbance are outlined in red) and one GIF per condition."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dp', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--env', default='visual-cube-single-v0')
    p.add_argument('--task', type=int, default=2)
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--conditions', default='none,push:0.08,noise:0.2,noise:0.4,burst:0.03')
    p.add_argument('--method', default='policy', help='method to drive the robot (needs --model for world models)')
    p.add_argument('--model', action='append', default=[])
    p.add_argument('--n-frames', type=int, default=8)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--n-samples', type=int, default=32)
    p.add_argument('--ddim-steps', type=int, default=16)
    p.add_argument('--exec-steps', type=int, default=8)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image

    from jepa.ogb_data import _env
    from jepa.ogb_eval import HiResCamera, Planner, run_episode
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device if (a.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    planner = Planner(a, dev)
    env = _env(a.env)
    env.reset(seed=a.seed)
    cam = HiResCamera(env, a.size, 'front')
    conds = a.conditions.split(',')
    n = a.n_frames
    fig, axes = plt.subplots(len(conds), n + 1, figsize=(1.6 * (n + 1), 1.8 * len(conds)), squeeze=False,
                             gridspec_kw={'wspace': 0.04, 'hspace': 0.3})
    seed = a.seed + 1000 * a.task + a.episode
    for r, cond in enumerate(conds):
        a.disturb = cond
        ok, steps, video, goal, events = run_episode(env, planner, a.method, a.task, seed, a, record=True,
                                                     cam=cam, return_events=True)
        print(f'{cond:12s} {a.method}: {"success" if ok else "fail"} in {steps} steps; disturbances at {events}')
        idx = np.linspace(0, len(video) - 1, n).round().astype(int)
        for e_ in events:  # make sure a frame right after each push / burst is shown
            if e_ + 1 < len(video) and not np.any(np.abs(idx - (e_ + 1)) <= 2):
                idx[np.argmin(np.abs(idx - (e_ + 1)))] = e_ + 1
        idx = np.sort(idx)
        for c_, i in enumerate(idx):
            ax = axes[r, c_]
            ax.imshow(video[i])
            ax.set_xticks([])
            ax.set_yticks([])
            hit = any(0 <= i - (e_ + 1) <= 4 for e_ in events)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color('#b8302f' if hit else '#d0d0d0')
                sp.set_linewidth(2.5 if hit else 0.5)
            last = i == idx[-1]
            ax.set_title(f'step {i}' + ((' success' if ok else ' fail') if last else ''), fontsize=8,
                         color=('#1baf7a' if ok else '#b8302f') if last else '#0b0b0b')
            if c_ == 0:
                ax.set_ylabel(cond.replace(':', '\n'), fontsize=9)
        ax = axes[r, n]
        ax.imshow(goal)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('goal image', fontsize=8)
        imgs = [Image.fromarray(f) for f in video[::2]]
        imgs[0].save(os.path.join(a.out, f'disturb_{cond.replace(":", "_")}.gif'), save_all=True,
                     append_images=imgs[1:], duration=70, loop=0)
    fig.suptitle(f'Disturbances during execution (task {a.task}, driven by: {a.method}; red = just disturbed)',
                 fontsize=10)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(a.out, f'fig_disturb_preview.{ext}'), dpi=200, bbox_inches='tight')
    print(f'[done] {os.path.join(a.out, "fig_disturb_preview.png")}')


if __name__ == '__main__':
    main()
