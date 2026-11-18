#!/usr/bin/env python
"""Train the Diffusion Policy on the cached dataset (train split; val split for monitoring).

    python -m jepa.dp_train --cache /mnt/t7shield/cache_jepa_128 --out runs/dp

Writes <out>/policy.pt (EMA weights, config, normalization stats; updated every epoch),
<out>/last.pt (for --resume) and <out>/log.csv.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from jepa.data import Cache, worker_init_fn
from jepa.models import torch_load
from jepa.policy import DiffusionPolicy, PolicyData


class FrameDataset(Dataset):
    def __init__(self, pdata, episodes, goal_cond=False, seed=0):
        c = pdata.c
        self.pd, self.goal_cond = pdata, goal_cond
        self.rng = np.random.RandomState(seed)
        self.frames = np.concatenate([np.arange(c.starts[e], c.starts[e] + c.lengths[e]) for e in episodes])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        t = self.frames[i:i + 1]
        pd, c = self.pd, self.pd.c
        f = pd.obs_frames(t)[0]
        img = np.ascontiguousarray(c.images[f]).transpose(0, 3, 1, 2)          # (2, 3, S, S) uint8
        out = {'img': torch.from_numpy(img), 'prop': torch.from_numpy(pd.proprio[f]),
               'task': int(c.episode_task[c.episode_of_frame[t[0]]]),
               'act': torch.from_numpy(pd.normalize(pd.actions[pd.action_rows(t)[0]]))}
        if self.goal_cond:
            g = pd.sample_goal_frames(t, self.rng)[0]
            out['goal'] = torch.from_numpy(np.array(c.images[g]).transpose(2, 0, 1).copy())
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=60000, help='optimizer steps (~100 epochs on 240 episodes)')
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--horizon', type=int, default=16, help='predicted per-step actions (10 Hz)')
    p.add_argument('--obs-skip', type=int, default=2, help='frames between the two observation steps')
    p.add_argument('--down-dims', default='128,256,512')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--device', default='cuda')
    p.add_argument('--amp', action='store_true', help='bfloat16 autocast')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--action-mode', default='delta_next', help="'stored' for OGBench (its recorded 5-d actions)")
    p.add_argument('--proprio-keys', default='q,dq,gripper')
    p.add_argument('--goal-cond', action='store_true', help='condition on a goal image (hindsight goals)')
    p.add_argument('--goal-p', type=float, default=0.01, help='hindsight goal offset ~ Geometric(p) frames')
    a = p.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(a.device if (a.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    os.makedirs(a.out, exist_ok=True)

    c = Cache(a.cache)
    tr_eps, va_eps = c.split_episodes('train'), c.split_episodes('val')
    pdata = PolicyData(c, a.horizon, a.obs_skip, a.proprio_keys.split(','), train_episodes=tr_eps,
                       action_mode=a.action_mode, goal_p=a.goal_p)
    train = DataLoader(FrameDataset(pdata, tr_eps, a.goal_cond, a.seed), batch_size=a.batch, shuffle=True, drop_last=True,
                       num_workers=a.workers, worker_init_fn=worker_init_fn, pin_memory=dev.type == 'cuda',
                       persistent_workers=a.workers > 0)
    val_set = FrameDataset(pdata, va_eps, a.goal_cond, 1)
    rng = np.random.RandomState(0)
    val_idx = rng.choice(len(val_set), min(2048, len(val_set)), replace=False)

    pol = DiffusionPolicy(n_tasks=len(c.tasks), proprio_dim=pdata.proprio.shape[1], horizon=a.horizon,
                          image_size=c.images.shape[1], action_dim=pdata.actions.shape[1], goal_cond=a.goal_cond,
                          down_dims=tuple(int(x) for x in a.down_dims.split(','))).to(dev)
    ema = copy.deepcopy(pol).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, weight_decay=1e-6)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    step = 0
    last = os.path.join(a.out, 'last.pt')
    if a.resume and os.path.exists(last):
        ck = torch_load(last, dev)
        pol.load_state_dict(ck['model'])
        ema.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        step = ck['step']
        print(f'resumed at step {step}')
    n_par = sum(q.numel() for q in pol.parameters()) / 1e6
    print(f'{len(train.dataset)} training frames ({len(tr_eps)} episodes), {len(val_set)} val frames; '
          f'{n_par:.1f}M parameters; device {dev}', flush=True)

    @torch.no_grad()
    def val_loss():
        g = torch.random.get_rng_state()
        torch.manual_seed(1234)  # same noise every time, so the curve is comparable
        tot = 0.0
        for i in range(0, len(val_idx), 256):
            items = [val_set[j] for j in val_idx[i:i + 256]]
            b = {k: (torch.stack([it[k] for it in items]) if k != 'task' else torch.tensor([it[k] for it in items]))
                 for k in items[0]}
            tot += float(ema.loss(b['img'].to(dev), b['prop'].to(dev), b['task'].to(dev), b['act'].to(dev),
                                  b['goal'].to(dev) if 'goal' in b else None)) * len(items)
        torch.random.set_rng_state(g)
        return tot / len(val_idx)

    def save():
        state = {'config': pol.cfg, 'stats': pdata.stats, 'tasks': list(c.tasks), 'obs_skip': a.obs_skip,
                 'step': step, 'action_mode': a.action_mode, 'goal_p': a.goal_p}
        torch.save(dict(state, ema=ema.state_dict()), os.path.join(a.out, 'policy.pt'))
        torch.save(dict(state, model=pol.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict()), last)

    log_path = os.path.join(a.out, 'log.csv')
    new_log = not (a.resume and os.path.exists(log_path))
    logf = open(log_path, 'w' if new_log else 'a', newline='')
    log = csv.writer(logf)
    if new_log:
        log.writerow(['step', 'epoch', 'train_loss', 'val_loss', 'lr', 'minutes'])
    t0, epoch, best = time.time(), step // max(len(train), 1), float('inf')
    while step < a.steps:
        pol.train()
        run, n = 0.0, 0
        for b in train:
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=a.amp):
                loss = pol.loss(b['img'].to(dev, non_blocking=True), b['prop'].to(dev, non_blocking=True),
                                b['task'].to(dev), b['act'].to(dev, non_blocking=True),
                                b['goal'].to(dev, non_blocking=True) if 'goal' in b else None)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            decay = min(0.9999, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), pol.parameters()):
                    pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
                for be, bm in zip(ema.buffers(), pol.buffers()):
                    be.copy_(bm)
            run += loss.item()
            n += 1
            if step % 100 == 0:
                print(f'step {step}/{a.steps}  loss {run / n:.4f}  lr {sched.get_last_lr()[0]:.2e}  '
                      f'{(time.time() - t0) / 60:.1f} min', flush=True)
            if step >= a.steps:
                break
        epoch += 1
        vl = val_loss()
        log.writerow([step, epoch, run / max(n, 1), vl, sched.get_last_lr()[0], (time.time() - t0) / 60])
        logf.flush()
        save()
        best = min(best, vl)
        print(f'== epoch {epoch}  step {step}  train {run / max(n, 1):.4f}  val {vl:.4f}  (best {best:.4f})',
              flush=True)
    logf.close()
    print(f'[done] {os.path.join(a.out, "policy.pt")}')


if __name__ == '__main__':
    main()
