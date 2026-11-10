#!/usr/bin/env python
"""Train a JEPA world model on the multi-task Franka cache.

    python -m jepa.train --cache cache_128 --config configs/msp_sigreg.yaml --save-dir runs/msp_sigreg

Outputs in --save-dir:
    config.yaml        resolved config (after `inherit`)
    history.json       per-epoch train/val terms, quick probes, lr
    model_best.pt      lowest validation total loss
    model_final.pt     last epoch (use this one for comparisons across objectives)
    checkpoints/       last.pt (for --resume) and epoch snapshots every checkpoint_every
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from jepa.data import ACTION_DIM, Cache, WindowDataset, action_stats, proprio_stats, worker_init_fn
from jepa.losses import compute_losses
from jepa.models import WorldModel, torch_load
from jepa.probes import ridge_classify, ridge_r2


def load_config(path):
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop('inherit', None)
    if parent is None:
        return cfg
    base = load_config(os.path.join(os.path.dirname(path), parent))

    def merge(a, b):
        out = copy.deepcopy(a)
        for k, v in b.items():
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        return out
    return merge(base, cfg)


def make_loaders(cache, cfg, batch_size, num_workers, seed):
    dc, mc = cfg['data'], cfg['model']
    tasks = dc.get('tasks')
    train_eps = cache.split_episodes('train', tasks)
    stats = action_stats(cache, dc['action_mode'], train_eps, dc.get('action_norm', 'scale'))
    pstats = None
    if mc.get('use_proprio', False):
        pstats = proprio_stats(cache, mc['proprio_keys'], train_eps)
        mc['proprio_dim'] = int(len(pstats['mean']))
    common = dict(horizon=dc['horizon'], frame_skip=dc['frame_skip'], action_mode=dc['action_mode'],
                  stats=stats, return_goal=bool(mc.get('use_gcb', False)),
                  goal_geom_p=dc.get('goal_geom_p', 0.3), goal_max_offset=dc.get('goal_max_offset', 30),
                  tasks=tasks, proprio=pstats)
    train = WindowDataset(cache, 'train', start_stride=dc.get('start_stride', 1), seed=seed, **common)
    val = WindowDataset(cache, 'val', start_stride=dc.get('val_start_stride', 4), seed=seed + 1, **common)
    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=num_workers > 0 and torch.cuda.is_available(),
              worker_init_fn=worker_init_fn, persistent_workers=num_workers > 0)
    return (DataLoader(train, shuffle=True, drop_last=True, **kw),
            # fixed random order: consecutive windows of one episode in a batch are highly
            # correlated, which inflates batch statistics such as SIGReg (val >> train)
            DataLoader(val, sampler=np.random.RandomState(seed).permutation(len(val)).tolist(), **kw),
            stats, pstats)


@torch.no_grad()
def quick_probe(model, train_loader, val_loader, cache, device, max_batches=40):
    """Ridge probes from z_t: EE position (R2), gripper (R2), task (accuracy)."""
    def collect(loader):
        zs, frames, tasks = [], [], []
        for i, b in enumerate(loader):
            if i >= max_batches:
                break
            prop = b['proprio'][:, 0].to(device) if 'proprio' in b else None
            zs.append(model.encode(b['obs'][:, 0].to(device), b['prev_obs'][:, 0].to(device), prop).float().cpu())
            frames.append(b['frame'][:, 0])
            tasks.append(b['task'])
        return torch.cat(zs).numpy(), torch.cat(frames).numpy(), torch.cat(tasks).numpy()
    model.eval()
    ztr, ftr, ttr = collect(train_loader)
    zva, fva, tva = collect(val_loader)
    pos, grip = cache.low['x_ee'][:, :3], cache.low['gripper']
    return {'r2_ee_pos': float(ridge_r2(ztr, pos[ftr], zva, pos[fva]).mean()),
            'r2_gripper': float(ridge_r2(ztr, grip[ftr], zva, grip[fva]).mean()),
            'task_acc': ridge_classify(ztr, ttr, zva, tva, len(cache.tasks))[0]}


@torch.no_grad()
def recalibrate_bn(model, loader, tc, device, n_batches=50, use_amp=False):
    """Recompute BatchNorm running statistics with the current weights ("precise BN").

    The pre-BN mean of the encoder/predictor projectors is invisible to every loss in
    train mode, so it drifts faster than the momentum-0.1 EMA can follow; without this
    the eval-mode encoder (used for validation, probes and planning) is badly off.
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [bn.momentum for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative average over the recalibration batches
    model.train()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            compute_losses(model, batch, tc, device)
    for bn, mom in zip(bns, momenta):
        bn.momentum = mom
    model.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--save-dir', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--num-workers', type=int)
    p.add_argument('--amp', action='store_true', help='bfloat16 autocast on CUDA')
    p.add_argument('--resume', action='store_true', help='continue from <save-dir>/checkpoints/last.pt')
    p.add_argument('--max-steps-per-epoch', type=int, default=None, help='debug: cap iterations')
    p.add_argument('--set', nargs='*', default=[], metavar='SECTION.KEY=VALUE',
                   help='override config entries, e.g. --set model.image_size=64 training.lr=3e-4')
    args = p.parse_args()

    cfg = load_config(args.config)
    for item in args.set:
        key, value = item.split('=', 1)
        section, name = key.split('.', 1)
        cfg.setdefault(section, {})[name] = yaml.safe_load(value)
    tc, dc, mc = cfg['training'], cfg['data'], cfg['model']
    for key, val in (('epochs', args.epochs), ('batch_size', args.batch_size),
                     ('num_workers', args.num_workers)):
        if val is not None:
            tc[key] = val
    mc['endpoint_inverse_horizon'] = dc['horizon']
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    with open(os.path.join(args.save_dir, 'config.yaml'), 'w') as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    cache = Cache(args.cache, dc.get('quat_order', 'wxyz'))
    if cache.meta['image_size'] != int(mc['image_size']):
        raise ValueError(f'cache images are {cache.meta["image_size"]}px but model.image_size='
                         f'{mc["image_size"]}; rebuild the cache or change the config')
    train_loader, val_loader, stats, pstats = make_loaders(cache, cfg, int(tc['batch_size']),
                                                           int(tc.get('num_workers', 4)), args.seed)
    if pstats is not None:
        print(f'[proprio] {pstats["keys"]} -> {mc["proprio_dim"]}-d, fused with the CLS feature')
    with open(os.path.join(args.save_dir, 'config.yaml'), 'w') as fh:  # now with proprio_dim
        yaml.safe_dump(cfg, fh, sort_keys=False)
    action_dim = ACTION_DIM * int(dc['frame_skip'])
    model = WorldModel(mc, action_dim).to(device)
    n_params = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f'[data] tasks={cache.tasks} train_windows={len(train_loader.dataset)} '
          f'val_windows={len(val_loader.dataset)} H={dc["horizon"]} frame_skip={dc["frame_skip"]} '
          f'action_dim={action_dim}')
    print(f'[model] {n_params / 1e6:.2f}M params, prediction={tc["prediction"]}, '
          + ', '.join(f'{k}={tc[k]}' for k in sorted(tc) if k.startswith('lambda_') and float(tc[k]) > 0))

    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=float(tc['lr']),
                            weight_decay=float(tc.get('weight_decay', 1e-3)))
    epochs = int(tc['epochs'])
    steps_per_epoch = min(len(train_loader), args.max_steps_per_epoch or len(train_loader))
    total_steps, warmup = epochs * steps_per_epoch, int(tc.get('warmup_epochs', 2)) * steps_per_epoch
    min_ratio = float(tc.get('min_lr', 1e-6)) / float(tc['lr'])

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start_epoch, best_val, history = 1, float('inf'), []
    last_path = os.path.join(args.save_dir, 'checkpoints', 'last.pt')
    if args.resume and os.path.exists(last_path):
        ck = torch_load(last_path, device)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler'])
        start_epoch, best_val, history = ck['epoch'] + 1, ck['best_val'], ck['history']
        print(f'[resume] from epoch {ck["epoch"]}')

    use_amp = args.amp and device.type == 'cuda'
    probe_every = int(tc.get('probe_every', 5))
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()
        sums, count = {}, 0
        for it, batch in enumerate(train_loader):
            if it >= steps_per_epoch:
                break
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                loss, terms, _ = compute_losses(model, batch, tc, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc.get('grad_clip', 1.0)))
            opt.step()
            sched.step()
            if not math.isfinite(terms['total']):
                raise RuntimeError(f'non-finite loss at epoch {epoch} iter {it}: {terms}')
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + v
            count += 1
        train_terms = {k: v / count for k, v in sums.items()}
        recalibrate_bn(model, train_loader, tc, device, int(tc.get('bn_recalib_batches', 50)), use_amp)

        model.eval()
        vsums, vcount = {}, 0
        with torch.no_grad():
            for batch in val_loader:
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    _, terms, _ = compute_losses(model, batch, tc, device, with_diagnostics=True)
                for k, v in terms.items():
                    vsums[k] = vsums.get(k, 0.0) + v
                vcount += 1
        val_terms = {k: v / vcount for k, v in vsums.items()}
        row = {'epoch': epoch, 'train': train_terms, 'val': val_terms, 'lr': sched.get_last_lr()[0],
               'time': time.time() - t0}
        if epoch % probe_every == 0 or epoch == epochs:
            row['probe'] = quick_probe(model, train_loader, val_loader, cache, device)
        history.append(row)
        msg = (f'[{epoch:3d}/{epochs}] ' + ' '.join(f'{k}={v:.4g}' for k, v in train_terms.items())
               + ' | val ' + ' '.join(f'{k}={val_terms[k]:.4g}' for k in
                                     ('total', 'err_one_step', 'err_multi_step', 'rel_err_multi_step', 'err_vs_copy', 'latent_std'))
               + (' | ' + ' '.join(f'{k}={v:.3f}' for k, v in row['probe'].items()) if 'probe' in row else '')
               + f' | {row["time"]:.0f}s')
        print(msg, flush=True)

        ckpt = {'model': model.state_dict(), 'config': cfg, 'action_dim': action_dim,
                'action_stats': {k: v.tolist() for k, v in stats.items()}, 'tasks': cache.tasks,
                'proprio_stats': None if pstats is None else
                {'keys': pstats['keys'], 'mean': pstats['mean'].tolist(), 'std': pstats['std'].tolist()},
                'epoch': epoch, 'val': val_terms}
        if val_terms['total'] < best_val:
            best_val = val_terms['total']
            torch.save(ckpt, os.path.join(args.save_dir, 'model_best.pt'))
        if epoch % int(tc.get('checkpoint_every', 10)) == 0:
            torch.save(ckpt, os.path.join(args.save_dir, 'checkpoints', f'epoch{epoch:04d}.pt'))
        torch.save(dict(ckpt, optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                        best_val=best_val, history=history), last_path)
        with open(os.path.join(args.save_dir, 'history.json'), 'w') as fh:
            json.dump(history, fh, indent=1)
    final = torch_load(last_path, 'cpu')
    for key in ('optimizer', 'scheduler', 'best_val', 'history'):
        final.pop(key)
    torch.save(final, os.path.join(args.save_dir, 'model_final.pt'))
    print(f'[done] {os.path.join(args.save_dir, "model_final.pt")}')


if __name__ == '__main__':
    main()
