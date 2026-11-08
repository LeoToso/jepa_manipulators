"""Training objectives.

Prediction (always Markov, z_{t+1} ~ f(z_t, a_t)):
    one_step (1SP)    teacher-forced: mean_k || f(z_k, a_k) - z_{k+1} ||^2
    multi_step (MSP)  autoregressive from z_0: mean_k || zhat_k - z_k ||^2,
                      zhat_{k+1} = f(zhat_k, a_k), gradients through the chain
    Targets are not stop-gradiented (as in JEPA_control / LeWorldModel); the
    anti-collapse term is what keeps the encoder from collapsing.

Anti-collapse / structure terms:
    SIGReg   LeJEPA (Balestriero & LeCun 2025), official Epps-Pulley form:
             N * int_{-3}^{3} |ecf(t) - exp(-t^2/2)|^2 exp(-t^2/2) dt, trapezoid rule
             on [0, 3] doubled by symmetry, averaged over random unit directions
             resampled every step. Applied to every latent in the window.
    EP-IDM   endpoint inverse dynamics (JEPA_control): the H normalized action
             chunks from (z_t, z_{t+H}) only, MSE.
    GCB      goal-conditioned bisimulation (Hansen-Estruch et al. 2022):
             L_phi = ( ||phi(s_i,g_i) - phi(s_j,g_j)||_1 - |r_i - r_j|
                       - gamma ||phibar(s'_i,g_i) - phibar(s'_j,g_j)||_2 )^2     (Eq. 6)
             L_psi = || (phibar(s,g) - phibar(g,g)) - (psi(g) - psi(s)) ||_2^2   (Eq. 7)
             (norms averaged per dimension by default, see gcb_losses)
             L_rew = BCE( R(phi(s,g), phi(s',g)), r )                             (App. D.5)
             with psi(s) = z (the JEPA latent), j a random permutation of the
             batch, s' the recorded next state and a reward-free hindsight reward:
             g is a later frame of the same episode and r = 1 iff s' = g.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sigreg_lejepa(z, num_slices=256, t_max=3.0, knots=17):
    z = z.float()
    n, d = z.shape
    t = torch.linspace(0, t_max, knots, device=z.device)
    dt = t_max / (knots - 1)
    w = torch.full((knots,), 2 * dt, device=z.device)
    w[[0, -1]] = dt
    phi = torch.exp(-0.5 * t ** 2)
    a = torch.randn(d, num_slices, device=z.device)
    a = a / a.norm(dim=0, keepdim=True)
    xt = (z @ a).unsqueeze(-1) * t                       # (N, slices, knots)
    err = (xt.cos().mean(0) - phi) ** 2 + xt.sin().mean(0) ** 2
    return ((err @ (w * phi)) * n).mean()


def sigreg_jepa_control(z, num_slices=128, num_points=17):
    """The variant used in JEPA_control (no weighting, no N factor), for reproducibility."""
    z = z.float()
    w = F.normalize(torch.randn(num_slices, z.shape[1], device=z.device), dim=1)
    y = w @ z.T
    omega = torch.linspace(0.5, 3.0, num_points, device=z.device)
    wy = omega[None, :, None] * y[:, None, :]
    target = torch.exp(-0.5 * omega ** 2)
    return ((wy.cos().mean(-1) - target) ** 2 + wy.sin().mean(-1) ** 2).mean()


def prediction_loss(model, z, actions, mode):
    """Returns (loss, predictions (b, H, d))."""
    b, t, d = z.shape
    if mode == 'one_step':
        pred = model.step(z[:, :-1].reshape(-1, d), actions.reshape(b * (t - 1), -1)).reshape(b, t - 1, d)
    elif mode == 'multi_step':
        pred = model.rollout(z[:, 0], actions)
    else:
        raise ValueError(f'prediction must be one_step or multi_step, got {mode!r}')
    return F.mse_loss(pred, z[:, 1:]), pred


def gcb_losses(model, z_s, z_next, z_goal, goal_offset, gamma=0.99, reward_scale=1.0,
               phi_grad_to_encoder=True, distance='normalized'):
    """distance='paper': L1 sum / L2 norm / squared-L2 sum exactly as in Eq. 6-7.
    distance='normalized' (default): the same equations with per-dimension averages
    (mean |.|, RMS, mean squares). The metric unit then matches a unit-scale latent for
    any latent_dim; with the raw sums the reward fixes distances of order 1 over all
    dimensions, and the joint optimum with the scale-dependent prediction MSE is to
    shrink z until prediction is free (observed: latent std ~0.004, useless predictor)."""
    heads = model.gcb
    d = z_s.shape[-1]
    if distance == 'paper':
        l1 = lambda x: x.abs().sum(-1)
        l2 = lambda x: x.norm(dim=-1)
        sq = lambda x: (x ** 2).sum(-1)
    elif distance == 'normalized':
        l1 = lambda x: x.abs().mean(-1)
        l2 = lambda x: x.norm(dim=-1) / d ** 0.5
        sq = lambda x: (x ** 2).mean(-1)
    else:
        raise ValueError(f'gcb_distance must be paper or normalized, got {distance!r}')
    r = (goal_offset == 1).float() * reward_scale
    zs, zg = (z_s, z_goal) if phi_grad_to_encoder else (z_s.detach(), z_goal.detach())
    phi = heads.pair(zs, zg)
    perm = torch.randperm(phi.shape[0], device=phi.device)
    dist = l1(phi - phi[perm])
    with torch.no_grad():
        phi_next = heads.pair(z_next.detach(), z_goal.detach())
        target = (r - r[perm]).abs() + gamma * l2(phi_next - phi_next[perm])
        anchor = heads.pair(z_s.detach(), z_goal.detach()) - heads.pair(z_goal.detach(), z_goal.detach())
    l_phi = ((dist - target) ** 2).mean()
    l_psi = sq(anchor - (z_goal - z_s)).mean()
    logits = heads.reward(torch.cat([phi, heads.pair(z_next if phi_grad_to_encoder else z_next.detach(), zg)],
                                    dim=-1))[:, 0]
    l_rew = F.binary_cross_entropy_with_logits(logits, (goal_offset == 1).float())
    return l_phi, l_psi, l_rew


def compute_losses(model, batch, tc, device, with_diagnostics=False):
    """Total loss and a dict of its terms (floats) for one batch."""
    obs = batch['obs'].to(device, non_blocking=True)
    prev = batch['prev_obs'].to(device, non_blocking=True)
    actions = batch['actions'].to(device, non_blocking=True).float()
    z = model.encode_seq(obs, prev)
    terms = {}
    l_pred, pred = prediction_loss(model, z, actions, tc['prediction'])
    total = float(tc.get('lambda_pred', 1.0)) * l_pred
    terms['pred'] = l_pred

    lam = float(tc.get('lambda_sigreg', 0.0))
    if lam > 0:
        zf = z.reshape(-1, z.shape[-1])
        if tc.get('sigreg_impl', 'lejepa') == 'lejepa':
            sig = sigreg_lejepa(zf, int(tc.get('sigreg_num_slices', 256)),
                                float(tc.get('sigreg_t_max', 3.0)), int(tc.get('sigreg_knots', 17)))
        else:
            sig = sigreg_jepa_control(zf, int(tc.get('sigreg_num_slices', 128)),
                                      int(tc.get('sigreg_knots', 17)))
        total = total + lam * sig
        terms['sigreg'] = sig

    lam = float(tc.get('lambda_epidm', 0.0))
    if lam > 0:
        a_hat = model.endpoint_inverse(z[:, 0], z[:, -1])
        inv = F.mse_loss(a_hat, actions)
        total = total + lam * inv
        terms['epidm'] = inv

    if model.gcb is not None and any(float(tc.get(k, 0.0)) > 0 for k in
                                     ('lambda_gcb_phi', 'lambda_gcb_psi', 'lambda_gcb_reward')):
        z_goal = model.encode(batch['goal_obs'].to(device, non_blocking=True),
                              batch['goal_prev_obs'].to(device, non_blocking=True))
        k = batch['goal_offset'].to(device)
        l_phi, l_psi, l_rew = gcb_losses(
            model, z[:, 0], z[:, 1], z_goal, k, float(tc.get('gcb_gamma', 0.99)),
            float(tc.get('gcb_reward_scale', 1.0)), bool(tc.get('gcb_phi_grad_to_encoder', True)),
            str(tc.get('gcb_distance', 'normalized')))
        total = (total + float(tc.get('lambda_gcb_phi', 1.0)) * l_phi
                 + float(tc.get('lambda_gcb_psi', 1.0)) * l_psi
                 + float(tc.get('lambda_gcb_reward', 1.0)) * l_rew)
        terms.update(gcb_phi=l_phi, gcb_psi=l_psi, gcb_reward=l_rew)

    terms['total'] = total
    if with_diagnostics:  # both prediction errors for every model, comparable across objectives
        with torch.no_grad():
            other = 'multi_step' if tc['prediction'] == 'one_step' else 'one_step'
            l_other, _ = prediction_loss(model, z, actions, other)
            terms[f'err_{tc["prediction"]}'] = l_pred.detach()
            terms[f'err_{other}'] = l_other
            # MSE per element / variance per dimension: 1.0 = no better than predicting the mean
            var = z.reshape(-1, z.shape[-1]).var(0).mean()
            terms['rel_err_multi_step'] = terms['err_multi_step'] / var.clamp_min(1e-12)
            terms['latent_std'] = z.reshape(-1, z.shape[-1]).std(0).mean()
    return total, {k: float(v.detach()) for k, v in terms.items()}, z
