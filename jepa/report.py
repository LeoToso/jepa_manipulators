#!/usr/bin/env python
"""One self-contained HTML page with every evaluation figure, side by side across models.

    python -m jepa.report --evals eval/msp_sigreg eval/msp_gcb ... --compare eval/compare --out eval/report.html

Each analysis is one row; each model is one column. The comparison table and the cmp_*.png
figures from jepa.compare come first. Images are embedded, so the file can be copied anywhere
(e.g. scp to a laptop) and opened in a browser.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os

SECTIONS = [  # (figure name, title, what to look for)
    ('probes', 'A. What the latent contains',
     'Linear vs. MLP R² of physical quantities, and task / family / object accuracy. Robot-state '
     'probes are partly fed by proprio; object identity comes from the images only.'),
    ('task_confusion', 'A. Task confusion',
     'Which tasks get mixed up: same family / different object, or across families?'),
    ('family_probes', 'B. Position vs. rotation per family',
     'Is one family\'s physics favoured? Compare EE position on pick-and-place with EE rotation on rotate.'),
    ('predicted_latents', 'B. What survives the predictor',
     'Black: error relative to the copy baseline (< 1 = uses the actions). Probes on k-step predictions vs. '
     'on the encoded target (dotted).'),
    ('spectrum', 'C. Dimensionality',
     'A flat spectrum and high effective rank = no dimensional collapse.'),
    ('gaussianity', 'C. Gaussianity (LeJEPA)',
     'Random 1-D projections of the latent vs. N(0,1).'),
    ('embedding_pca', 'C. PCA of test latents', 'Coloured by task, progress, gripper, EE height.'),
    ('embedding_tsne', 'C. t-SNE of test latents',
     'Do tasks form separate clusters, or does robot state (gripper, height) dominate?'),
    ('within_task_pca', 'C. Within-task PCA (BeJEPA)',
     'Task means removed: do the leading directions follow EE x / y / z, rotation, or only the gripper?'),
    ('task_subspaces', 'C. Shared vs. task-specific directions',
     'Overlap of per-task principal subspaces (1 = the same directions).'),
    ('temporal', 'C. Latent trajectories', 'Straightening and transition-subspace dimension per task.'),
    ('goal_distance', 'D. Goal structure',
     'Latent distance to the final frame along episodes (ideal: monotone to 0).'),
    ('analogies', 'D. Analogies (GCB)',
     'z(B_0) + z(A_T) − z(A_0) → nearest neighbour: does it land at an end state (progress near 1)?'),
    ('successor', 'D. Successor features',
     'Successor models only: goal value along episodes (ideal: rising) and far-goal planning.'),
    ('planning', 'E. Offline CEM planning',
     'P(random beats recorded) near 0 = a good cost landscape; action NMSE = how close CEM gets to the demo.'),
    ('patch_pca', 'F. What the encoder looks at', 'First 3 principal components of the patch tokens as RGB.'),
]

COMPARE_FIGS = ['cmp_probes', 'cmp_position_vs_rotation', 'cmp_predicted', 'cmp_geometry',
                'cmp_temporal', 'cmp_goal', 'cmp_planning', 'cmp_training', 'cmp_loss_vs_probe']


def _img(path):
    if not os.path.exists(path):
        return '<div class="missing">not available</div>'
    with open(path, 'rb') as fh:
        data = base64.b64encode(fh.read()).decode()
    return f'<img src="data:image/png;base64,{data}" loading="lazy">'


def _table(md_path):
    if not os.path.exists(md_path):
        return ''
    rows = [r.strip().strip('|').split('|') for r in open(md_path).read().strip().splitlines()]
    head, body = rows[0], [r for r in rows[2:]]
    out = ['<table><thead><tr>' + ''.join(f'<th>{html.escape(c.strip())}</th>' for c in head) + '</tr></thead><tbody>']
    for r in body:
        out.append('<tr>' + ''.join(f'<td>{html.escape(c.strip())}</td>' for c in r) + '</tr>')
    return ''.join(out) + '</tbody></table>'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--evals', nargs='+', required=True)
    p.add_argument('--compare', default=None, help='folder written by jepa.compare')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    evals = [d for d in a.evals if os.path.exists(os.path.join(d, 'metrics.json'))]
    names = []
    for d in evals:
        with open(os.path.join(d, 'metrics.json')) as fh:
            names.append(json.load(fh).get('name') or os.path.basename(os.path.normpath(d)))

    parts = ['<!doctype html><html><head><meta charset="utf-8"><title>JEPA latent report</title>',
             '<meta name="viewport" content="width=device-width, initial-scale=1">', '<style>',
             'body{font-family:-apple-system,system-ui,sans-serif;margin:16px;background:#fff;color:#111}',
             'h1{font-size:22px}h2{font-size:17px;margin-top:28px;border-bottom:1px solid #ddd}',
             'p.note{color:#555;font-size:13px;max-width:1100px}',
             '.row{display:flex;gap:10px;overflow-x:auto;padding-bottom:6px}',
             '.cell{flex:0 0 auto;width:min(46vw,760px)}.cell h3{font-size:13px;margin:4px 0}',
             '.cell img{width:100%;border:1px solid #eee}.missing{color:#999;font-size:12px;padding:20px}',
             'table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:3px 6px}',
             'th{background:#f4f4f4;position:sticky;top:0}td:not(:first-child){text-align:right}',
             '.wide img{max-width:100%;border:1px solid #eee}nav a{margin-right:10px;font-size:13px}',
             '@media (prefers-color-scheme: dark){body{background:#111;color:#eee}th{background:#222}',
             'td,th{border-color:#333}p.note{color:#aaa}img{background:#fff}}',
             '</style></head><body>',
             f'<h1>JEPA latent-space report: {", ".join(html.escape(n) for n in names)}</h1>']
    nav = ['<nav>', '<a href="#summary">summary</a>'] + [f'<a href="#{s[0]}">{html.escape(s[1])}</a>' for s in SECTIONS] + ['</nav>']
    parts += nav
    if a.compare:
        parts.append('<h2 id="summary">Summary table</h2>')
        parts.append(_table(os.path.join(a.compare, 'summary.md')))
        parts.append('<h2>Comparison figures</h2>')
        for f in COMPARE_FIGS:
            path = os.path.join(a.compare, f + '.png')
            if os.path.exists(path):
                parts.append(f'<div class="wide"><h3>{f}</h3>{_img(path)}</div>')
    for fig, title, note in SECTIONS:
        if not any(os.path.exists(os.path.join(d, 'figs', fig + '.png')) for d in evals):
            continue
        parts.append(f'<h2 id="{fig}">{html.escape(title)}</h2><p class="note">{html.escape(note)}</p><div class="row">')
        for d, n in zip(evals, names):
            parts.append(f'<div class="cell"><h3>{html.escape(n)}</h3>{_img(os.path.join(d, "figs", fig + ".png"))}</div>')
        parts.append('</div>')
    parts.append('</body></html>')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as fh:
        fh.write('\n'.join(parts))
    print(f'[done] {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB, {len(evals)} models)')


if __name__ == '__main__':
    main()
