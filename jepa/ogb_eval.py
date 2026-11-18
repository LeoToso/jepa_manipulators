#!/usr/bin/env python
"""Closed-loop planning evaluation in OGBench: goal-conditioned Diffusion Policy alone vs. steered by
each world model, measured by OGBench's own success criterion.

    MUJOCO_GL=egl python -m jepa.ogb_eval --env visual-cube-single-v0 --dp runs/ogb/dp --out eval/ogb \\
        --model "MSP+SIG=runs/ogb/msp_sigreg" --model "MSP+GCB+SIG=runs/ogb/msp_gcb_sigreg" \\
        --model "MSP+SIG+SF=runs/ogb/msp_sigreg_sf"

For each of OGBench's 5 evaluation tasks and --episodes-per-task episodes (the same seeds for every
method), the robot gets the task's goal image. Every --exec-steps environment steps it re-plans:
the policy proposes N action chunks of 16 steps conditioned on the current images, proprio and the
goal image; a selection method picks one; its first --exec-steps actions are executed. Methods:
    policy        a single policy sample (the baseline)
    consensus     the candidate closest to all others (choosing among N without a world model)
    <model>       the candidate whose predicted latent after the chunk is closest to the goal latent
    <model> (value)  successor models only: the candidate with the highest goal value
An episode succeeds when OGBench reports success (all cubes at their goals) within the episode limit.
Writes results.json, summary.md, fig_success.png/pdf and (--videos) one GIF per method and task.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from jepa.models import load_checkpoint  # noqa: E402
from jepa.ogb_data import _env, proprio_from_info  # noqa: E402
from jepa.policy import load_policy  # noqa: E402

MODEL_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False, 'figure.facecolor': 'white',
    'savefig.facecolor': 'white', 'pdf.fonttype': 42,
})


class Planner:
    """Policy candidates + selection by one method."""

    def __init__(self, a, dev):
        self.dev = dev
        self.pol, ck = load_policy(a.dp if a.dp.endswith('.pt') else os.path.join(a.dp, 'policy.pt'), dev)
        if not self.pol.goal_cond:
            raise SystemExit('the policy must be goal-conditioned (train it with --goal-cond)')
        st = ck['stats']
        self.a_min, self.a_max = np.array(st['a_min'], np.float32), np.array(st['a_max'], np.float32)
        self.p_mean, self.p_std = np.array(st['p_mean'], np.float32), np.array(st['p_std'], np.float32)
        self.obs_skip, self.H, self.N, self.ddim = int(ck['obs_skip']), self.pol.horizon, a.n_samples, a.ddim_steps
        self.models = {}
        for spec in a.model:
            label, run = spec.split('=', 1)
            m, wck = load_checkpoint(run if run.endswith('.pt') else os.path.join(run, 'model_final.pt'), dev)
            m.eval()
            dc = wck['config']['data']
            if dc['action_mode'] != 'stored' or self.H % int(dc['frame_skip']):
                raise SystemExit(f'{label}: needs action_mode stored and frame_skip dividing {self.H}')
            self.models[label] = dict(model=m, fs=int(dc['frame_skip']), stats=wck['action_stats'],
                                      proprio=wck.get('proprio_stats'),
                                      sf=getattr(m, 'successor', None) is not None)

    def methods(self):
        out = ['policy', 'consensus']
        for label, m in self.models.items():
            out.append(label)
            if m['sf']:
                out.append(f'{label} (value)')
        return out

    @torch.no_grad()
    def candidates(self, frames, props, goal, gen):
        t = len(frames) - 1
        idx = [max(t - self.obs_skip, 0), t]
        img = torch.from_numpy(np.stack([frames[i] for i in idx])).permute(0, 3, 1, 2)[None].to(self.dev)
        prop = torch.from_numpy(np.stack([(props[i] - self.p_mean) / self.p_std for i in idx]))[None].to(self.dev)
        g = torch.from_numpy(goal).permute(2, 0, 1)[None].to(self.dev)
        cond = self.pol.encode_obs(img, prop.float(), torch.zeros(1, dtype=torch.long, device=self.dev), g)
        x = self.pol.sample(cond.repeat_interleave(self.N, 0), self.ddim, generator=gen).cpu().numpy()
        return x, ((x + 1) / 2 * (self.a_max - self.a_min) + self.a_min).astype(np.float32)

    @torch.no_grad()
    def select(self, method, cand_n, cand, frames, props, goal):
        if method == 'policy':
            return 0
        flat = cand_n.reshape(len(cand_n), -1)
        if method == 'consensus':
            return int(((flat[:, None] - flat[None]) ** 2).mean(-1).sum(-1).argmin())
        label = method.replace(' (value)', '')
        m = self.models[label]
        model, fs = m['model'], m['fs']
        t = len(frames) - 1

        def img(x):
            return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].to(self.dev)
        prop = None
        if m['proprio'] is not None:  # a proprio-fused world model: current proprio for z_t
            ps = m['proprio']
            prop = torch.from_numpy(((props[t] - np.array(ps['mean'])) / np.array(ps['std'])).astype(np.float32))[None].to(self.dev)
        z0 = model.encode(img(frames[t]), img(frames[max(t - fs, 0)]), prop)
        zg = model.encode(img(goal), img(goal), prop)  # a goal image has no motion: prev = goal
        st = m['stats']
        act = (cand - np.array(st['mean'], np.float32)) / np.array(st['std'], np.float32)
        act = torch.from_numpy(act.reshape(len(cand), self.H // fs, -1).astype(np.float32)).to(self.dev)
        zend = model.rollout(z0.expand(len(cand), -1), act)[:, -1]
        if method.endswith('(value)'):
            return int(model.goal_value(zend, zg.expand_as(zend)).argmax())
        return int(((zend - zg) ** 2).mean(-1).argmin())


class HiResCamera:
    """Renders the scene at a higher resolution for videos only (the agent still sees 64 x 64)."""

    def __init__(self, env, size, camera):
        import mujoco
        u = env.unwrapped
        u._model.vis.global_.offwidth = max(u._model.vis.global_.offwidth, size)
        u._model.vis.global_.offheight = max(u._model.vis.global_.offheight, size)
        self.r, self.u, self.camera = mujoco.Renderer(u._model, size, size), u, camera

    def __call__(self):
        self.r.update_scene(self.u._data, camera=self.camera)
        return self.r.render().copy()


def run_episode(env, planner, method, task_id, seed, a, record=False, cam=None):
    ob, info = env.reset(seed=seed, options=dict(task_id=task_id, render_goal=False))
    goal = info['goal']
    frames, props = [ob], [proprio_from_info(info)]
    snap = (lambda o: cam()) if cam is not None else (lambda o: o)
    video = [snap(ob)] if record else None
    gen = torch.Generator(device=planner.dev).manual_seed(seed)
    max_steps = a.max_steps or env.spec.max_episode_steps
    t, success = 0, False
    while t < max_steps and not success:
        cand_n, cand = planner.candidates(frames, props, goal, gen)
        k = planner.select(method, cand_n, cand, frames, props, goal)
        for action in cand[k][:a.exec_steps]:
            ob, _, terminated, truncated, info = env.step(np.clip(action, -1, 1))
            frames.append(ob)
            props.append(proprio_from_info(info))
            if record:
                video.append(snap(ob))
            t += 1
            if info.get('success', False):
                success = True
            if terminated or truncated or success or t >= max_steps:
                break
    return success, t, video, goal


def save_gif(frames, goal, path):
    """Episode next to its goal image (the goal is the 64 x 64 image the agent receives)."""
    from PIL import Image
    h = frames[0].shape[0]
    g = np.asarray(Image.fromarray(goal).resize((h, h), Image.NEAREST))
    size = (2 * max(h, 128), max(h, 128))
    imgs = [Image.fromarray(np.concatenate([f, g], axis=1)).resize(size, Image.NEAREST) for f in frames[::2]]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=80, loop=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env', default='visual-cube-single-v0')
    p.add_argument('--dp', required=True)
    p.add_argument('--model', action='append', default=[], help='LABEL=RUN_DIR (RUN_DIR/model_final.pt)')
    p.add_argument('--out', required=True)
    p.add_argument('--tasks', default='1,2,3,4,5')
    p.add_argument('--episodes-per-task', type=int, default=10)
    p.add_argument('--n-samples', type=int, default=32)
    p.add_argument('--ddim-steps', type=int, default=16)
    p.add_argument('--exec-steps', type=int, default=8, help='actions executed before re-planning')
    p.add_argument('--max-steps', type=int, default=0, help='0 = the environment limit')
    p.add_argument('--methods', default='', help='comma-separated subset (default: all)')
    p.add_argument('--videos', action='store_true', help='one GIF per method and task (first episode)')
    p.add_argument('--video-size', type=int, default=256, help='video resolution (render only; 64 = what the agent sees)')
    p.add_argument('--video-camera', default='front', help="'front' (opaque arm) or 'front_pixels' (the agent's view)")
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device if (a.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    planner = Planner(a, dev)
    env = _env(a.env)
    cam = None
    methods = [m for m in planner.methods() if not a.methods or m in a.methods.split(',')]
    tasks = [int(x) for x in a.tasks.split(',')]
    res_path = os.path.join(a.out, 'results.json')
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for method in methods:
        for task in tasks:
            key = f'{method}|task{task}'
            if key in res and len(res[key]['success']) >= a.episodes_per_task:
                continue  # already done (resumable)
            succ, steps = [], []
            t0 = time.time()
            for e in range(a.episodes_per_task):
                seed = a.seed + 1000 * task + e  # the same start states for every method
                rec = a.videos and e == 0
                if rec and cam is None and a.video_size != 64:
                    env.reset(seed=seed)
                    cam = HiResCamera(env, a.video_size, a.video_camera)
                ok, n, video, goal = run_episode(env, planner, method, task, seed, a, record=rec, cam=cam if rec else None)
                succ.append(bool(ok))
                steps.append(int(n))
                if video is not None:
                    save_gif(video, goal, os.path.join(a.out, f'{method.replace(" ", "_").replace("+", "")}_task{task}.gif'))
            res[key] = {'success': succ, 'steps': steps}
            with open(res_path, 'w') as fh:
                json.dump(res, fh, indent=1)
            print(f'{method:28s} task {task}: success {np.mean(succ):.2f} ({sum(succ)}/{len(succ)})  '
                  f'{time.time() - t0:.0f}s', flush=True)
    report(res, methods, tasks, list(planner.models), a.out, a.env)


def report(res, methods, tasks, models, out, env_name):
    rows, summ = [], {}
    for m in methods:
        per = [np.mean(res[f'{m}|task{t}']['success']) for t in tasks]
        allv = np.concatenate([res[f'{m}|task{t}']['success'] for t in tasks]).astype(float)
        se = allv.std(ddof=1) / np.sqrt(len(allv)) if len(allv) > 1 else 0.0
        summ[m] = {'per_task': per, 'overall': float(allv.mean()), 'se': float(se), 'n': int(len(allv))}
        rows.append(f'| {m} | ' + ' | '.join(f'{100 * v:.0f}' for v in per) + f' | **{100 * allv.mean():.1f}** ± {100 * se:.1f} |')
    head = '| method | ' + ' | '.join(f'task {t}' for t in tasks) + ' | overall (%) |'
    md = [f'{env_name}: success rate (%), {summ[methods[0]]["n"] // len(tasks)} episodes per task', '',
          head, '|' + '---|' * (len(tasks) + 2)] + rows
    open(os.path.join(out, 'summary.md'), 'w').write('\n'.join(md) + '\n')
    with open(os.path.join(out, 'summary.json'), 'w') as fh:
        json.dump(summ, fh, indent=1)
    print('\n'.join(md))

    def style(m):
        if m == 'policy':
            return INK, None
        if m == 'consensus':
            return INK2, None
        base = m.replace(' (value)', '')
        col = MODEL_COLORS[models.index(base) % 3] if base in models else INK2
        return col, ('//' if m.endswith('(value)') else None)
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * (len(tasks) + 1) * len(methods) / 3), 3.6))
    groups = [f'task {t}' for t in tasks] + ['overall']
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        col, hatch = style(m)
        vals = summ[m]['per_task'] + [summ[m]['overall']]
        xs = np.arange(len(groups)) - 0.4 + w * (j + 0.5)
        ax.bar(xs, 100 * np.array(vals), w * 0.92, color=col, hatch=hatch, edgecolor='white', linewidth=0.5,
               label=m if m not in ('policy', 'consensus') else ('policy alone' if m == 'policy' else 'consensus of N'))
        ax.errorbar(xs[-1], 100 * vals[-1], yerr=100 * summ[m]['se'], color=INK, capsize=2, lw=1)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel('success rate [%]')
    ax.set_ylim(0, 100)
    ax.grid(axis='x', visible=False)
    ax.legend(fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    ax.set_title(env_name)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'fig_success.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[done] {out}')


if __name__ == '__main__':
    main()
