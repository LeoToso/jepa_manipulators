"""Figures for jepa.evaluate (one model) — see jepa.compare for cross-model figures."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TASK_COLORS = plt.cm.tab10(np.arange(10))
LABELS = {'ee_pos': 'EE position', 'ee_rot': 'EE rotation', 'gripper': 'gripper', 'joints': 'joint angles',
          'ee_lin_vel': 'EE lin. vel.', 'ee_ang_vel': 'EE ang. vel.', 'progress': 'task progress'}


def _save(fig, figs, name):
    fig.savefig(os.path.join(figs, name + '.png'), dpi=150, bbox_inches='tight')
    fig.savefig(os.path.join(figs, name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def _title(M, text):
    return f'{M["name"]}: {text}'


def probes(M, figs):
    A = M['A']
    qs = list(A['linear_r2'])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={'width_ratios': [2.2, 1]})
    x = np.arange(len(qs))
    a1.bar(x - 0.2, [A['linear_r2'][q] for q in qs], 0.4, label='linear (ridge)')
    mlp = [A['mlp_r2'].get(q) for q in qs]
    a1.bar(x + 0.2, [m if m is not None else 0 for m in mlp], 0.4, label='MLP')
    a1.set_xticks(x, [LABELS[q] for q in qs], rotation=25, ha='right')
    a1.set_ylabel('test R$^2$')
    a1.set_ylim(min(0, min(A['linear_r2'].values())) - 0.05, 1.02)
    a1.axhline(0, color='k', lw=0.6)
    a1.legend(fontsize=8)
    a1.set_title('physical quantities from $z_t$')
    keys = ['task', 'family', 'object']
    x = np.arange(len(keys))
    a2.bar(x - 0.2, [A[f'{k}_acc_linear'] for k in keys], 0.4, label='linear')
    a2.bar(x + 0.2, [A[f'{k}_acc_knn'] for k in keys], 0.4, label='kNN (k=20)')
    a2.set_xticks(x, keys)
    a2.set_ylim(0, 1.02)
    a2.set_ylabel('test accuracy')
    a2.legend(fontsize=8)
    a2.set_title(f'identity (action R$^2$ from $z_t,z_{{t+1}}$ = {A["action_r2_linear"]:.2f})', fontsize=9)
    fig.suptitle(_title(M, 'what the encoded latent contains'))
    _save(fig, figs, 'probes')

    conf = np.array(A['task_confusion'])
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(conf, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(M['tasks'])), M['tasks'], rotation=40, ha='right', fontsize=7)
    ax.set_yticks(range(len(M['tasks'])), M['tasks'], fontsize=7)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    for i in range(len(conf)):
        for j in range(len(conf)):
            ax.text(j, i, f'{conf[i, j]:.2f}', ha='center', va='center', fontsize=6,
                    color='white' if conf[i, j] > 0.5 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(_title(M, 'linear task probe confusion'), fontsize=9)
    _save(fig, figs, 'task_confusion')


def family_probes(M, figs):
    B = M['B_family']
    fams = M['families']
    qs = ['ee_pos', 'ee_rot', 'gripper', 'ee_lin_vel', 'ee_ang_vel']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, suffix, title in ((axes[0], '', 'global probe, evaluated per family'),
                              (axes[1], '_within', 'probe fitted inside the family')):
        x = np.arange(len(qs))
        w = 0.8 / len(fams)
        for i, f in enumerate(fams):
            ax.bar(x + (i - (len(fams) - 1) / 2) * w, [B[f + suffix][q] for q in qs], w, label=f)
        ax.set_xticks(x, [LABELS[q] for q in qs], rotation=20, ha='right')
        ax.axhline(0, color='k', lw=0.6)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('test R$^2$ within family')
    fig.suptitle(_title(M, 'is position favoured over rotation? (per task family)'))
    _save(fig, figs, 'family_probes')


def predicted(M, figs):
    P = M['B_predicted']
    k = P['k']
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    axes[0].plot(k, P['rel_err'], 'o-', label='all')
    for fam, v in P['per_family_rel_err'].items():
        axes[0].plot(k, v, '--', label=fam)
    if 'err_vs_copy' in P:
        axes[0].plot(k, P['err_vs_copy'], 's-', color='k', label='vs copy baseline z_t')
        axes[0].axhline(1, color='k', lw=0.6, ls=':')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('prediction horizon k (latent steps)')
    axes[0].set_ylabel('MSE / latent variance  |  MSE / copy MSE')
    axes[0].set_title('rollout error (black: < 1 beats "nothing changes")', fontsize=9)
    axes[0].legend(fontsize=8)
    for i, q in enumerate(['ee_pos', 'ee_rot', 'gripper', 'progress']):
        axes[1].plot(k, P['r2'][q], 'o-', color=TASK_COLORS[i], label=LABELS[q])
        axes[1].plot(k, P['r2_encoded_target'][q], ':', color=TASK_COLORS[i])
    axes[1].set_ylim(-0.5, 1.02)  # R^2 is unbounded below; values under -0.5 are clipped
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('R$^2$ of probe on $\\hat z_{t+k}$')
    axes[1].set_title('physical info in predictions (dotted: encoded target)', fontsize=9)
    axes[1].legend(fontsize=7)
    axes[2].plot(k, P['task_acc'], 'o-')
    axes[2].set_ylim(0, 1.02)
    axes[2].set_xlabel('k')
    axes[2].set_ylabel('task accuracy from $\\hat z_{t+k}$')
    axes[2].set_title('task identity in predictions')
    fig.suptitle(_title(M, 'what survives the predictor'))
    _save(fig, figs, 'predicted_latents')


def spectrum(M, figs):
    G = M['C_geometry']
    ev = np.array(G['eigenvalues'])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    a1.semilogy(np.arange(1, len(ev) + 1), np.clip(ev, 1e-12, None))
    a1.set_xlabel('component')
    a1.set_ylabel('normalized eigenvalue')
    a1.set_title(f'covariance spectrum (eff. rank {G["effective_rank"]:.1f}, '
                 f'PR {G["participation_ratio"]:.1f}, d={len(ev)})', fontsize=9)
    tasks = list(G['effective_rank_per_task'])
    a2.bar(range(len(tasks)), [G['effective_rank_per_task'][t] for t in tasks], color=TASK_COLORS[:len(tasks)])
    a2.set_xticks(range(len(tasks)), tasks, rotation=30, ha='right', fontsize=8)
    a2.set_ylabel('effective rank')
    a2.set_title('per-task effective rank')
    fig.suptitle(_title(M, 'dimensionality of the latent space'))
    _save(fig, figs, 'spectrum')


def gaussianity(zn, figs, rng, n_dirs=6):
    fig, axes = plt.subplots(1, n_dirs, figsize=(2.3 * n_dirs, 2.4), sharey=True)
    xs = np.linspace(-4, 4, 200)
    for ax in axes:
        a = rng.randn(zn.shape[1])
        a /= np.linalg.norm(a)
        y = zn @ a
        ax.hist(y, bins=60, range=(-4, 4), density=True, alpha=0.7)
        ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), 'k', lw=1)
        ax.set_yticks([])
    fig.suptitle('random 1-D projections of the centred, RMS-normalized latent vs N(0,1)', fontsize=10)
    _save(fig, figs, 'gaussianity')


def embeddings(z, task, lab, frames, tasks, figs, rng, tsne=True, n=4000):
    idx = rng.choice(len(z), min(n, len(z)), replace=False)
    zc = z[idx] - z[idx].mean(0)
    pca = zc @ np.linalg.svd(zc, full_matrices=False)[2][:2].T
    views = [('PCA', pca)]
    if tsne:
        try:
            from sklearn.manifold import TSNE
            views.append(('t-SNE', TSNE(2, init='pca', random_state=0, perplexity=30).fit_transform(zc)))
        except ImportError:
            pass
    colorings = [('task', task[idx], None), ('progress', lab['progress'][frames[idx], 0], 'viridis'),
                 ('gripper', lab['gripper'][frames[idx], 0], 'coolwarm'),
                 ('EE height', lab['ee_pos'][frames[idx], 2], 'magma')]
    for name, emb in views:
        fig, axes = plt.subplots(1, 4, figsize=(17, 4))
        for ax, (cname, c, cmap) in zip(axes, colorings):
            if cmap is None:
                for i, t in enumerate(tasks):
                    s = c == i
                    ax.scatter(emb[s, 0], emb[s, 1], s=3, color=TASK_COLORS[i], label=t)
                ax.legend(fontsize=6, markerscale=3)
            else:
                sc = ax.scatter(emb[:, 0], emb[:, 1], s=3, c=c, cmap=cmap)
                plt.colorbar(sc, ax=ax, fraction=0.046)
            ax.set_title(cname)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f'{name} of test latents')
        _save(fig, figs, f'embedding_{name.lower().replace("-", "")}')


def task_subspaces(M, figs):
    G = M['C_geometry']
    o = np.array(G['task_subspace_overlap'])
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(o, vmin=0, vmax=1, cmap='viridis')
    ax.set_xticks(range(len(M['tasks'])), M['tasks'], rotation=40, ha='right', fontsize=7)
    ax.set_yticks(range(len(M['tasks'])), M['tasks'], fontsize=7)
    for i in range(len(o)):
        for j in range(len(o)):
            ax.text(j, i, f'{o[i, j]:.2f}', ha='center', va='center', fontsize=6, color='w' if o[i, j] < .6 else 'k')
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f'{M["name"]}: overlap of top-{G["subspace_dim"]} task subspaces\n'
                 f'(between/within task variance = {G["between_within_task_ratio"]:.2f})', fontsize=9)
    _save(fig, figs, 'task_subspaces')


def temporal(M, figs):
    T = M['C_geometry']['temporal']
    tasks = list(T)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, key, ylabel in ((axes[0], 'straightening', 'mean cos(dz_t, dz_{t+1})'),
                            (axes[1], 'r95', 'transition r95 (components)')):
        ax.bar(range(len(tasks)), [T[t][key][0] for t in tasks], yerr=[T[t][key][1] for t in tasks],
               color=TASK_COLORS[:len(tasks)], capsize=3)
        ax.set_xticks(range(len(tasks)), tasks, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel(ylabel)
    axes[0].set_title('temporal straightening')
    axes[1].set_title('transition subspace dimension')
    fig.suptitle(_title(M, 'latent trajectories'))
    _save(fig, figs, 'temporal')


def goal_distance(M, figs):
    D = M['D_goal']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    for i, (t, curve) in enumerate(D['curves'].items()):
        a1.plot(D['grid'], curve, color=TASK_COLORS[i], label=t)
    a1.set_xlabel('episode progress')
    a1.set_ylabel('||z_t - z_goal|| / ||z_0 - z_goal||')
    a1.set_title('latent distance to the final frame (ideal: monotone to 0)', fontsize=9)
    a1.legend(fontsize=7)
    tasks = list(D['spearman_mean'])
    a2.bar(range(len(tasks)), [D['spearman_mean'][t] for t in tasks], color=TASK_COLORS[:len(tasks)])
    a2.set_xticks(range(len(tasks)), tasks, rotation=30, ha='right', fontsize=8)
    a2.set_ylim(-1, 1)
    a2.axhline(0, color='k', lw=.6)
    a2.set_ylabel('Spearman(distance, remaining time)')
    extra = f' | steps-to-goal probe R$^2$={D["steps_to_goal_r2"]:.2f}' if 'steps_to_goal_r2' in D else ''
    a2.set_title('planning-cost monotonicity' + extra, fontsize=9)
    fig.suptitle(_title(M, 'goal structure'))
    _save(fig, figs, 'goal_distance')


def analogies(M, figs):
    An = M['D_goal'].get('analogies', {})
    if not An:
        return
    keys = ['retrieved_progress', 'retrieved_task_is_B', 'retrieved_same_episode_as_B']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(len(keys))
    for i, (kind, v) in enumerate(An.items()):
        a1.bar(x + (i - 0.5) * 0.4, [v[k] for k in keys], 0.4, label=kind.replace('_', ' '))
        a2.bar([i - 0.2], [v['ee_pos_error_to_B_end']], 0.35, color='C0')
        a2.bar([i + 0.2], [v['baseline_ee_pos_error']], 0.35, color='C7')
    a1.set_xticks(x, ['progress of NN', 'NN has task of B', 'NN from episode B'], fontsize=8)
    a1.set_ylim(0, 1.02)
    a1.legend(fontsize=8)
    a1.set_title('z(B_0) + z(A_T) - z(A_0) -> nearest neighbour', fontsize=9)
    a2.set_xticks(range(len(An)), [k.replace('_', ' ') for k in An], fontsize=8)
    a2.set_ylabel('EE distance to B_T [m]')
    a2.set_title('blue: analogy NN, grey: start of B (baseline)', fontsize=9)
    fig.suptitle(_title(M, 'latent analogies'))
    _save(fig, figs, 'analogies')


def planning(M, figs):
    E = M['E_planning']
    keys = ['cost_ratio', 'gt_rank', 'action_nmse', 'first_action_nmse', 'gripper_agree']
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(len(keys)), [E[k] for k in keys])
    ax.set_xticks(range(len(keys)), ['CEM cost / GT cost', 'P(random < GT)', 'action NMSE',
                                     '1st action NMSE', 'gripper sign agree'], rotation=15, fontsize=8)
    ax.set_title(_title(M, f'offline CEM to the frame {E["horizon"]} latent steps ahead ({E["trials"]} trials)'),
                 fontsize=9)
    _save(fig, figs, 'planning')


def patch_pca(images, tokens, names, figs):
    n, p, c = tokens.shape
    g = int(round(np.sqrt(p)))
    flat = tokens.reshape(-1, c)
    flat = flat - flat.mean(0)
    proj = flat @ np.linalg.svd(flat, full_matrices=False)[2][:3].T
    lo, hi = np.percentile(proj, 1, axis=0), np.percentile(proj, 99, axis=0)
    rgb = np.clip((proj - lo) / np.maximum(hi - lo, 1e-8), 0, 1).reshape(n, g, g, 3)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    axes = np.array(axes).reshape(2, n)
    for i in range(n):
        axes[0, i].imshow(images[i])
        axes[0, i].set_title(names[i], fontsize=7)
        axes[1, i].imshow(rgb[i], interpolation='nearest')
        for ax in axes[:, i]:
            ax.axis('off')
    fig.suptitle('encoder patch tokens: first 3 principal components as RGB', fontsize=10)
    _save(fig, figs, 'patch_pca')
