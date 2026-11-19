#!/usr/bin/env python
"""Side-by-side videos of several methods on the same OGBench episode (same start, same goal).

    MUJOCO_GL=egl python -m jepa.ogb_video --dp runs/ogb/dp --eval-dir eval/ogb_cube_single \\
        --methods "policy,consensus,MSP+SIG" --model "MSP+SIG=runs/ogb/msp_sigreg" --tasks 1,2,3,4,5

For each task, the episode is chosen from the evaluation's results.json: the first episode where
--prefer succeeded (default: the first method), else episode 0. Its seed is the evaluation's seed,
so the start state, goal and policy samples are the ones evaluated. Each method's run is rendered
at --size, side by side with the goal image, with the method name and SUCCESS / FAIL on top.
Writes <eval-dir>/videos/compare_task<k>_ep<e>.gif.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from jepa.ogb_data import _env
from jepa.ogb_eval import HiResCamera, Planner, run_episode


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1: scalable default font
    except TypeError:
        try:
            return ImageFont.truetype('DejaVuSans.ttf', size)
        except OSError:
            return ImageFont.load_default()


def _centered(d, x0, width, y, text, font, fill):
    w = d.textbbox((0, 0), text, font=font)[2]
    d.text((x0 + (width - w) / 2, y), text, font=font, fill=fill)


PRETTY = {'policy': 'Policy alone', 'consensus': 'Consensus of N'}


def compose(videos, labels, oks, goal, size, path):
    """Panels side by side, each with its title centred above it: the method on the first line,
    the time step / SUCCESS / FAIL on the second; the goal image last."""
    T = max(len(v) for v in videos)
    g = Image.fromarray(goal).resize((size, size), Image.NEAREST)
    fs = max(12, size // 16)
    f_title, f_status = _font(fs), _font(int(fs * 0.9))
    bar = int(2.6 * fs) + 6
    frames = []
    for t in range(0, T, 2):
        canvas = Image.new('RGB', (size * (len(videos) + 1), size + bar), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        for j, (v, lab, ok) in enumerate(zip(videos, labels, oks)):
            done = t >= len(v) - 1
            canvas.paste(Image.fromarray(v[min(t, len(v) - 1)]), (j * size, bar))
            status = ('SUCCESS' if ok else 'FAIL') if done else f'step {t}'
            col = (27, 175, 122) if (done and ok) else ((184, 48, 47) if done else (82, 81, 78))
            _centered(d, j * size, size, 3, PRETTY.get(lab, lab), f_title, (11, 11, 11))
            _centered(d, j * size, size, 3 + int(1.3 * fs), status, f_status, col)
        canvas.paste(g, (len(videos) * size, bar))
        _centered(d, len(videos) * size, size, 3, 'Goal image', f_title, (11, 11, 11))
        _centered(d, len(videos) * size, size, 3 + int(1.3 * fs), '(what the agent sees)', f_status, (82, 81, 78))
        frames.append(canvas)
    frames += [frames[-1]] * 10  # hold the end
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=70, loop=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env', default='visual-cube-single-v0')
    p.add_argument('--dp', required=True)
    p.add_argument('--model', action='append', default=[], help='LABEL=RUN_DIR, for world-model methods')
    p.add_argument('--methods', required=True, help='comma-separated, e.g. "policy,consensus,MSP+SIG"')
    p.add_argument('--eval-dir', required=True, help='folder of jepa.ogb_eval (results.json, seeds)')
    p.add_argument('--tasks', default='1,2,3,4,5')
    p.add_argument('--episode', type=int, default=-1, help='fixed episode index (default: chosen from results)')
    p.add_argument('--prefer', default='', help='pick an episode where this method succeeded (default: first)')
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--camera', default='front')
    p.add_argument('--n-samples', type=int, default=32)
    p.add_argument('--ddim-steps', type=int, default=16)
    p.add_argument('--exec-steps', type=int, default=8)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--seed', type=int, default=0, help='the evaluation seed (jepa.ogb_eval --seed)')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    dev = torch.device(a.device if (a.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    methods = [m.strip() for m in a.methods.split(',')]
    planner = Planner(a, dev)
    env = _env(a.env)
    env.reset(seed=a.seed)
    cam = HiResCamera(env, a.size, a.camera)
    res_path = os.path.join(a.eval_dir, 'results.json')
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    out = os.path.join(a.eval_dir, 'videos')
    os.makedirs(out, exist_ok=True)
    prefer = a.prefer or methods[0]
    for task in [int(x) for x in a.tasks.split(',')]:
        e = a.episode
        if e < 0:
            succ = res.get(f'{prefer}|task{task}', {}).get('success', [])
            e = int(np.argmax(succ)) if any(succ) else 0
        seed = a.seed + 1000 * task + e
        videos, oks, goal = [], [], None
        for m in methods:
            ok, n, video, goal = run_episode(env, planner, m, task, seed, a, record=True, cam=cam)
            videos.append(video)
            oks.append(bool(ok))
            print(f'task {task} episode {e}: {m:24s} {"success" if ok else "fail"} in {n} steps', flush=True)
        path = os.path.join(out, f'compare_task{task}_ep{e}.gif')
        compose(videos, methods, oks, goal, a.size, path)
        print(f'  wrote {path}')


if __name__ == '__main__':
    main()
