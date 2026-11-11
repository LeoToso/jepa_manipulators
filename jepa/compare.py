#!/usr/bin/env python
"""Compare evaluated models side by side.

    python -m jepa.compare --evals eval/1sp_sigreg eval/msp_sigreg eval/1sp_epidm \\
        eval/msp_epidm eval/1sp_gcb eval/msp_gcb --runs runs/1sp_sigreg ... --out eval/compare

--evals are folders written by jepa.evaluate (metrics.json); --runs (optional) are
training folders (history.json) for training curves and the loss-vs-probe plot.

Writes <out>/summary.md, summary.csv and figures:
    cmp_probes.png              linear R^2 per quantity, task/family/object accuracy
    cmp_position_vs_rotation.png  position info on pick-and-place vs rotation info on rotate
    cmp_predicted.png           rollout error and probe R^2 on k-step predictions
    cmp_geometry.png            spectra, effective rank, task separation, Gaussianity
    cmp_temporal.png            straightening and transition r95 per family
    cmp_goal.png                distance-to-goal monotonicity, steps-to-goal probe, analogies
    cmp_planning.png            offline CEM metrics
    cmp_training.png            validation curves and quick probes over epochs (needs --runs)
    cmp_loss_vs_probe.png       training loss vs probe across epochs (LeJEPA Fig. 1 style)
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jepa.data import task_family  # noqa: E402
from jepa.plots import LABELS  # noqa: E402

COLORS = plt.cm.tab10(np.arange(10))


def _save(fig, out, name):
    fig.savefig(os.path.join(out, name + '.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(out, name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def _bars(ax, groups, series, names, ylabel=None, ylim=None):
    x = np.arange(len(groups))
    w = 0.8 / len(series)
    for i, (vals, name) in enumerate(zip(series, names)):
        ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, label=name, color=COLORS[i % 10])
    ax.set_xticks(x, groups, rotation=20, ha='right', fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.axhline(0, color='k', lw=0.5)


def family_mean(per_task, fam):
    vals = [v for t, v in per_task.items() if task_family(t) == fam]
    return float(np.mean(vals)) if vals else float('nan')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--evals', nargs='+', required=True)
    p.add_argument('--runs', nargs='*', default=[])
    p.add_argument('--names', nargs='*', default=None)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Ms = []
    for d in a.evals:
        path = os.path.join(d, 'metrics.json')
        if not os.path.exists(path):  # e.g. the output folder itself when using eval/*
            print(f'[skip] {d}: no metrics.json')
            continue
        with open(path) as fh:
            Ms.append(json.load(fh))
    a.runs = [r for r in a.runs if os.path.exists(os.path.join(r, 'history.json'))]
    names = a.names or [m['name'] for m in Ms]
    fams = Ms[0]['families']
    quantities = list(Ms[0]['A']['linear_r2'])

    # ---- summary table ----------------------------------------------------------
    cols = [('task acc (lin)', lambda m: m['A']['task_acc_linear']),
            ('task acc (kNN)', lambda m: m['A']['task_acc_knn']),
            ('family acc', lambda m: m['A']['family_acc_linear']),
            ('object acc', lambda m: m['A']['object_acc_linear'])]
    cols += [(f'R2 {LABELS[q]}', (lambda q: lambda m: m['A']['linear_r2'][q])(q)) for q in quantities]
    cols += [('action R2 (z_t,z_t+1)', lambda m: m['A']['action_r2_linear'])]
    for f in fams:
        cols += [(f'R2 pos | {f}', (lambda f: lambda m: m['B_family'][f + '_within']['ee_pos'])(f)),
                 (f'R2 rot | {f}', (lambda f: lambda m: m['B_family'][f + '_within']['ee_rot'])(f))]
    cols += [('rollout rel err k=1', lambda m: m['B_predicted']['rel_err'][0]),
             ('rollout rel err k=K', lambda m: m['B_predicted']['rel_err'][-1]),
             ('rollout err / copy k=1', lambda m: m['B_predicted'].get('err_vs_copy', [float('nan')])[0]),
             ('rollout err / copy k=K', lambda m: m['B_predicted'].get('err_vs_copy', [float('nan')])[-1]),
             ('R2 EE pos at k=K', lambda m: m['B_predicted']['r2']['ee_pos'][-1]),
             ('task acc at k=K', lambda m: m['B_predicted']['task_acc'][-1]),
             ('effective rank', lambda m: m['C_geometry']['effective_rank']),
             ('between/within task var', lambda m: m['C_geometry']['between_within_task_ratio']),
             ('Epps-Pulley (RMS-norm.)', lambda m: m['C_geometry']['epps_pulley_rms_normalized']),
             ('goal-distance Spearman', lambda m: float(np.mean(list(m['D_goal']['spearman_mean'].values())))),
             ('steps-to-goal R2', lambda m: m['D_goal'].get('steps_to_goal_r2', float('nan'))),
             ('analogy NN progress', lambda m: m['D_goal'].get('analogies', {}).get('same_task', {}).get('retrieved_progress', float('nan'))),
             ('within-task PC1-3 R2 EE z', lambda m: m.get('C_within_task_pca', {}).get('r2_from_pc123', {}).get('EE z', float('nan'))),
             ('within-task PC1-3 R2 EE rot z', lambda m: m.get('C_within_task_pca', {}).get('r2_from_pc123', {}).get('EE rot z', float('nan'))),
             ('value Spearman (SF)', lambda m: float(np.mean(list(m['SF']['value_spearman_mean'].values()))) if 'SF' in m else float('nan')),
             ('far-goal 1st NMSE, value (SF)', lambda m: m['SF']['far_goal_planning']['value']['first_action_nmse'] if 'SF' in m else float('nan')),
             ('far-goal 1st NMSE, distance (SF)', lambda m: m['SF']['far_goal_planning']['distance']['first_action_nmse'] if 'SF' in m else float('nan')),
             ('CEM action NMSE', lambda m: m['E_planning']['action_nmse']),
             ('P(random < GT cost)', lambda m: m['E_planning']['gt_rank'])]
    table = [[fn(m) for _, fn in cols] for m in Ms]
    with open(os.path.join(a.out, 'summary.csv'), 'w') as fh:
        fh.write('model,' + ','.join(c for c, _ in cols) + '\n')
        for n, row in zip(names, table):
            fh.write(n + ',' + ','.join(f'{v:.4f}' for v in row) + '\n')
    with open(os.path.join(a.out, 'summary.md'), 'w') as fh:
        fh.write('| metric | ' + ' | '.join(names) + ' |\n|---|' + '---|' * len(names) + '\n')
        for j, (c, _) in enumerate(cols):
            fh.write(f'| {c} | ' + ' | '.join(f'{row[j]:.3f}' for row in table) + ' |\n')

    # ---- A: probes ----------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={'width_ratios': [2.3, 1]})
    _bars(a1, [LABELS[q] for q in quantities], [[m['A']['linear_r2'][q] for q in quantities] for m in Ms],
          names, 'test R$^2$ (linear)', (-0.1, 1.02))
    a1.legend(fontsize=7, ncol=2)
    a1.set_title('physical information in $z_t$')
    keys = ['task_acc_linear', 'task_acc_knn', 'family_acc_linear', 'object_acc_linear']
    _bars(a2, ['task (lin)', 'task (kNN)', 'family', 'object'], [[m['A'][k] for k in keys] for m in Ms],
          names, 'test accuracy', (0, 1.02))
    a2.set_title('task identity in $z_t$')
    _save(fig, a.out, 'cmp_probes')

    # ---- B: position vs rotation ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    groups, series = [], [[] for _ in Ms]
    for f in fams:
        for q in ('ee_pos', 'ee_rot', 'ee_lin_vel', 'ee_ang_vel'):
            groups.append(f'{LABELS[q]} | {f}')
            for i, m in enumerate(Ms):
                series[i].append(m['B_family'][f + '_within'][q])
    _bars(axes[0], groups, series, names, 'test R$^2$ (probe fitted within family)', (-0.1, 1.02))
    axes[0].legend(fontsize=7)
    axes[0].set_title('which physical quantities each family keeps')
    if 'pick_and_place' in fams and 'rotate' in fams:
        for i, (m, n) in enumerate(zip(Ms, names)):
            axes[1].scatter(m['B_family']['pick_and_place_within']['ee_pos'],
                            m['B_family']['rotate_within']['ee_rot'], s=60, color=COLORS[i], label=n)
        axes[1].plot([0, 1], [0, 1], 'k:', lw=0.8)
        axes[1].set_xlabel('R$^2$ EE position | pick-and-place')
        axes[1].set_ylabel('R$^2$ EE rotation | rotate')
        axes[1].set_title('balance between the two families (diagonal = equal)')
        axes[1].legend(fontsize=7)
    _save(fig, a.out, 'cmp_position_vs_rotation')

    # ---- B: predicted latents --------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for i, (m, n) in enumerate(zip(Ms, names)):
        P = m['B_predicted']
        axes[0].plot(P['k'], P['rel_err'], 'o-', color=COLORS[i], label=n, ms=3)
        if 'err_vs_copy' in P:
            axes[0].plot(P['k'], P['err_vs_copy'], 's--', color=COLORS[i], ms=3)
        axes[1].plot(P['k'], P['r2']['ee_pos'], 'o-', color=COLORS[i], ms=3)
        axes[2].plot(P['k'], P['r2']['ee_rot'], 'o-', color=COLORS[i], ms=3)
        axes[3].plot(P['k'], P['task_acc'], 'o-', color=COLORS[i], ms=3)
    axes[0].set_yscale('log')
    axes[0].axhline(1, color='k', lw=0.6, ls=':')
    for ax, t in zip(axes, ['rollout MSE / var (solid), / copy baseline (dashed)', 'R$^2$ EE position from $\\hat z_{t+k}$',
                            'R$^2$ EE rotation from $\\hat z_{t+k}$', 'task acc. from $\\hat z_{t+k}$']):
        ax.set_title(t, fontsize=9)
        ax.set_xlabel('k')
    for ax in axes[1:3]:
        ax.set_ylim(-0.5, 1.02)
    axes[3].set_ylim(0, 1.02)
    axes[0].legend(fontsize=7)
    _save(fig, a.out, 'cmp_predicted')

    # ---- C: geometry ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for i, (m, n) in enumerate(zip(Ms, names)):
        ev = np.clip(np.array(m['C_geometry']['eigenvalues']), 1e-12, None)
        axes[0].semilogy(np.arange(1, len(ev) + 1), ev, color=COLORS[i], label=n)
    axes[0].set_title('normalized covariance spectrum')
    axes[0].set_xlabel('component')
    axes[0].legend(fontsize=7)
    for ax, key, title in ((axes[1], 'effective_rank', 'effective rank'),
                           (axes[2], 'between_within_task_ratio', 'between / within task variance'),
                           (axes[3], 'epps_pulley_rms_normalized', 'Epps-Pulley (lower = more Gaussian)')):
        ax.bar(range(len(Ms)), [m['C_geometry'][key] for m in Ms], color=COLORS[:len(Ms)])
        ax.set_xticks(range(len(Ms)), names, rotation=30, ha='right', fontsize=8)
        ax.set_title(title)
    _save(fig, a.out, 'cmp_geometry')

    # ---- C: temporal -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, key in ((axes[0], 'straightening'), (axes[1], 'r95')):
        series = [[family_mean({t: v[key][0] for t, v in m['C_geometry']['temporal'].items()}, f)
                   for f in fams] for m in Ms]
        _bars(ax, fams, series, names, key)
        ax.set_title({'straightening': 'temporal straightening', 'r95': 'transition subspace r95'}[key])
    axes[0].legend(fontsize=7)
    _save(fig, a.out, 'cmp_temporal')

    # ---- D: goal structure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
    _bars(axes[0], fams, [[family_mean(m['D_goal']['spearman_mean'], f) for f in fams] for m in Ms],
          names, 'Spearman(distance to goal, remaining time)', (-1, 1))
    axes[0].set_title('planning-cost monotonicity')
    axes[0].legend(fontsize=7)
    axes[1].bar(range(len(Ms)), [m['D_goal'].get('steps_to_goal_r2', np.nan) for m in Ms], color=COLORS[:len(Ms)])
    axes[1].set_xticks(range(len(Ms)), names, rotation=30, ha='right', fontsize=8)
    axes[1].set_title('steps-to-goal probe R$^2$ from $(z_t, z_g)$')
    kinds = ['same_task', 'same_family_other_object']
    series = [[m['D_goal'].get('analogies', {}).get(k, {}).get('retrieved_progress', np.nan) for k in kinds]
              for m in Ms]
    _bars(axes[2], ['same task', 'same family, other object'], series, names, 'progress of retrieved NN', (0, 1.02))
    axes[2].set_title('analogy z(B_0)+z(A_T)-z(A_0): does it land at an end state?', fontsize=9)
    _save(fig, a.out, 'cmp_goal')

    # ---- E: planning ----------------------------------------------------------------------
    keys = ['cost_ratio', 'gt_rank', 'action_nmse', 'first_action_nmse', 'gripper_agree']
    fig, ax = plt.subplots(figsize=(11, 3.6))
    _bars(ax, ['CEM / GT cost', 'P(random < GT)', 'action NMSE', '1st action NMSE', 'gripper agree'],
          [[m['E_planning'][k] for k in keys] for m in Ms], names)
    ax.legend(fontsize=7)
    ax.set_title('offline CEM towards a recorded goal frame')
    _save(fig, a.out, 'cmp_planning')

    # ---- training curves / loss vs probe ----------------------------------------------------
    if a.runs:
        hists = []
        for r in a.runs:
            with open(os.path.join(r, 'history.json')) as fh:
                hists.append(json.load(fh))
        run_names = [os.path.basename(os.path.normpath(r)) for r in a.runs]
        # same colour as the model's evaluation (matched by folder name) in every figure
        rc = [COLORS[names.index(n) % 10] if n in names else COLORS[(len(names) + i) % 10]
              for i, n in enumerate(run_names)]
        fig, axes = plt.subplots(1, 4, figsize=(18, 3.6))
        for i, (h, n) in enumerate(zip(hists, run_names)):
            ep = [r['epoch'] for r in h]
            axes[0].plot(ep, [r['val']['rel_err_multi_step'] for r in h], color=rc[i], label=n)
            axes[1].plot(ep, [r['val']['err_one_step'] / max(r['val']['latent_std'] ** 2, 1e-12) for r in h],
                         color=rc[i])
            pe = [r for r in h if 'probe' in r]
            axes[2].plot([r['epoch'] for r in pe], [r['probe']['r2_ee_pos'] for r in pe], 'o-', color=rc[i])
            axes[3].plot([r['epoch'] for r in pe], [r['probe']['task_acc'] for r in pe], 'o-', color=rc[i])
        for ax, t in zip(axes, ['val rollout MSE / var (H steps)', 'val one-step MSE / std$^2$',
                                'quick probe R$^2$ EE position', 'quick probe task accuracy']):
            ax.set_title(t, fontsize=9)
            ax.set_xlabel('epoch')
        axes[0].set_yscale('log')
        axes[1].set_yscale('log')
        axes[0].legend(fontsize=7)
        _save(fig, a.out, 'cmp_training')

        fig, axes = plt.subplots(1, len(hists), figsize=(3.4 * len(hists), 3.2), squeeze=False)
        for ax, h, n in zip(axes[0], hists, run_names):
            pe = [r for r in h if 'probe' in r]
            x = [r['train']['total'] for r in pe]
            y = [r['probe']['r2_ee_pos'] for r in pe]
            sc = ax.scatter(x, y, c=[r['epoch'] for r in pe], cmap='viridis')
            if len(pe) > 2:
                rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
                ax.set_title(f'{n}\nSpearman = {np.corrcoef(rx, ry)[0, 1]:.2f}', fontsize=9)
            else:
                ax.set_title(n, fontsize=9)
            ax.set_xlabel('train loss (own objective)')
            ax.set_ylabel('probe R$^2$ EE position')
        fig.colorbar(sc, ax=axes[0].tolist(), label='epoch', fraction=0.02)
        _save(fig, a.out, 'cmp_loss_vs_probe')
    print(f'[done] {a.out}/summary.md and figures')


if __name__ == '__main__':
    main()
