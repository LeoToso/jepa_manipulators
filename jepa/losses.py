"""Training objectives.

Prediction (as JEPA_control):
    one_step (1SP)    teacher-forced over the window with the causal predictor:
                      mean_k || f(z_0..z_k, a_0..a_k) - z_{k+1} ||^2  (one_step_context: markov
                      uses f(z_k, a_k) only)
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
             L_phi = ( ||phi(s_i,g_i) - phi(s_j,g_j)||_1 - (1-gamma) |r_i - r_j|
                       - gamma ||phibar(s'_i,g_i) - phibar(s'_j,g_j)||_2 )^2     (Eq. 6 with
                                                         the (1-gamma) of their Prop. 6.1)
             L_psi = || (phibar(s,g) - phibar(g,g)) - (psi(g) - psi(s)) ||_2^2   (Eq. 7)
             (default: per-dimension mean |.| on both sides, see gcb_losses)
             L_rew = MSE( R(phi(s,g), phi(s',g)), r )  (BCE for the 0/1 indicator)  (App. D.5)
             with psi(s) = z (the JEPA latent), phi with its own image encoder (as in
             the paper; phi never depends on psi), j a random permutation of the batch, s' the recorded next state and a reward-free hindsight reward:
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


def prediction_loss(model, z, actions, mode, one_step_context='window'):
    """Returns (loss, predictions (b, H, d)).

    one_step_context='window' (JEPA_control): teacher-forced over the whole window with the
        causal predictor, so the prediction of z_{t+1} attends to z_0..z_t.
    one_step_context='markov': each z_{t+1} predicted from z_t alone.
    """
    b, t, d = z.shape
    if mode == 'one_step' and one_step_context == 'window':
        pred = model.predict_window(z[:, :-1], actions)
    elif mode == 'one_step':
        pred = model.step(z[:, :-1].reshape(-1, d), actions.reshape(b * (t - 1), -1)).reshape(b, t - 1, d)
    elif mode == 'multi_step':
        pred = model.rollout(z[:, 0], actions)
    else:
        raise ValueError(f'prediction must be one_step or multi_step, got {mode!r}')
    return F.mse_loss(pred, z[:, 1:]), pred


def gcb_reward(kind, goal_offset, next_state=None, goal_state=None, z_next=None, z_goal=None,
               state_scales=(0.05, 0.5, 2.0)):
    """Reward-free goal-reaching reward r(s, a, g), evaluated at the next state s'.

    indicator       1[s' = g] (the paper's sparse reward). With hindsight goals on logged,
                    deterministic trajectories the bisimulation fixed point then depends only
                    on the steps-to-goal k: d = (gamma^(k_i-1) + gamma^(k_j-1)) 1[k_i != k_j],
                    so it cannot tell tasks, objects or motions apart.
    state_distance  -(|p' - p_g| / s_p + angle(q', q_g) / s_q + |grip' - grip_g| / s_g) / 3 from
                    measured EE pose and gripper (no task reward; dense; rotation enters directly).
    latent_distance -mean |zbar' - zbar_g| with the (stop-gradient) JEPA latent: purely
                    self-supervised but self-referential.
    """
    if kind == 'indicator':
        return (goal_offset == 1).float()
    if kind == 'state_distance':
        dpos = (next_state[:, :3] - goal_state[:, :3]).norm(dim=-1)
        dot = (next_state[:, 3:7] * goal_state[:, 3:7]).sum(-1).abs().clamp(max=1.0)
        drot = 2 * torch.acos(dot)
        dgrip = (next_state[:, 7] - goal_state[:, 7]).abs()
        sp, sq, sg = state_scales
        return -(dpos / sp + drot / sq + dgrip / sg) / 3
    if kind == 'latent_distance':
        return -(z_next.detach() - z_goal.detach()).abs().mean(-1)
    raise ValueError(f'gcb_reward must be indicator, state_distance or latent_distance, got {kind!r}')


def gcb_losses(model, x_s, x_next, x_goal, z_s, z_goal, r, binary_reward, gamma=0.99,
               distance='normalized', proprio=None):
    """x_*: encoder inputs of s, s', g (same pass, so s' = g gives identical features);
    z_s, z_goal: JEPA latents psi(s), psi(g); r: reward of each sample (see gcb_reward).

    distance='paper': exactly Eq. 6-7: L1 sum for the current pair, L2 norm for the
        bootstrapped next pair, squared-L2 sum in the analogy loss. Since L1 >= L2 the
        update contracts, but the reward fixes distances of order 1 summed over all
        dimensions; with the scale-dependent prediction MSE the joint optimum is to shrink
        z until prediction is free (observed: latent std 0.004, useless predictor).
    distance='normalized' (default): per-dimension mean |.| on BOTH sides and a mean
        square in the analogy loss. Using the same norm on both sides makes the update a
        gamma-contraction (unique fixed point, their Prop. 6.1); per-dimension averaging
        makes the metric unit match a unit-scale latent for any latent_dim. (Mixing
        mean |.| with RMS is NOT a contraction, mean|x| <= RMS(x), and diverged.)"""
    heads = model.gcb
    if distance == 'paper':
        l1 = lambda x: x.abs().sum(-1)
        l2 = lambda x: x.norm(dim=-1)
        sq = lambda x: (x ** 2).sum(-1)
    elif distance == 'normalized':
        l1 = lambda x: x.abs().mean(-1)
        l2 = l1
        sq = lambda x: (x ** 2).mean(-1)
    else:
        raise ValueError(f'gcb_distance must be paper or normalized, got {distance!r}')
    b = x_s.shape[0]
    f_s, f_n, f_g = heads.encoder(torch.cat([x_s, x_next, x_goal]), proprio).split(b)
    phi = heads.pair(f_s, f_g)
    phi_next = heads.pair(f_n, f_g)
    perm = torch.randperm(b, device=phi.device)
    dist = l1(phi - phi[perm])
    with torch.no_grad():
        # (1 - gamma) weighting as in their Prop. 6.1, F = (1-c)|dR| + c W(d): same fixed-point
        # geometry, scaled to O(max |dr|). Eq. 6 drops it; since consecutive frames give
        # phi(s', g) ~ phi(s, g), the unweighted fixed point is ~|dr| / (1 - gamma) = 100x, and
        # the slowly inflating metric dragged psi (and the prediction loss) with it.
        target = (1 - gamma) * (r - r[perm]).abs() + gamma * l2(phi_next - phi_next[perm])
        anchor = phi - heads.pair(f_g, f_g)
    l_phi = ((dist - target) ** 2).mean()
    l_psi = sq(anchor - (z_goal - z_s)).mean()
    out = heads.reward(torch.cat([phi, phi_next], dim=-1))[:, 0]
    l_rew = (F.binary_cross_entropy_with_logits(out, (r > 0).float()) if binary_reward
             else F.mse_loss(out, r))
    return l_phi, l_psi, l_rew


def compute_losses(model, batch, tc, device, with_diagnostics=False):
    """Total loss and a dict of its terms (floats) for one batch."""
    obs = batch['obs'].to(device, non_blocking=True)
    prev = batch['prev_obs'].to(device, non_blocking=True)
    actions = batch['actions'].to(device, non_blocking=True).float()
    proprio = batch['proprio'].to(device, non_blocking=True) if model.use_proprio else None
    use_gcb = model.gcb is not None and any(float(tc.get(k, 0.0)) > 0 for k in
                                            ('lambda_gcb_phi', 'lambda_gcb_psi', 'lambda_gcb_reward'))
    b, t = obs.shape[:2]
    if use_gcb:
        # Goals go through the encoder in the same forward pass as the window: in train mode
        # BatchNorm uses per-call statistics, so a separate call would give the same frame a
        # different latent and break both r = 1[s' = g] and the phi(g, g) anchor.
        flat_obs = torch.cat([obs.reshape(b * t, *obs.shape[2:]), batch['goal_obs'].to(device, non_blocking=True)])
        flat_prev = torch.cat([prev.reshape(b * t, *prev.shape[2:]),
                               batch['goal_prev_obs'].to(device, non_blocking=True)])
        flat_prop = None
        if proprio is not None:
            flat_prop = torch.cat([proprio.reshape(b * t, -1), batch['goal_proprio'].to(device, non_blocking=True)])
        zall = model.encode(flat_obs, flat_prev, flat_prop)
        z, z_goal = zall[:b * t].reshape(b, t, -1), zall[b * t:]
    else:
        z = model.encode_seq(obs, prev, proprio)
    terms = {}
    ctx = str(tc.get('one_step_context', 'window'))
    l_pred, pred = prediction_loss(model, z, actions, tc['prediction'], ctx)
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

    if use_gcb:
        k = batch['goal_offset'].to(device)
        x_goal = model.encoder_input(batch['goal_obs'].to(device, non_blocking=True),
                                     batch['goal_prev_obs'].to(device, non_blocking=True))
        kind = str(tc.get('gcb_reward', 'state_distance'))
        r = gcb_reward(kind, k, batch.get('next_state', torch.zeros(1)).to(device),
                       batch.get('goal_state', torch.zeros(1)).to(device), z[:, 1], z_goal,
                       tuple(tc.get('gcb_state_scales', (0.05, 0.5, 2.0))))
        r = r * float(tc.get('gcb_reward_scale', 1.0))
        l_phi, l_psi, l_rew = gcb_losses(
            model, model.encoder_input(obs[:, 0], prev[:, 0]), model.encoder_input(obs[:, 1], prev[:, 1]),
            x_goal, z[:, 0], z_goal, r, kind == 'indicator', float(tc.get('gcb_gamma', 0.99)),
            str(tc.get('gcb_distance', 'normalized')),
            None if proprio is None else torch.cat([proprio[:, 0], proprio[:, 1],
                                                    batch['goal_proprio'].to(device, non_blocking=True)]))
        total = (total + float(tc.get('lambda_gcb_phi', 1.0)) * l_phi
                 + float(tc.get('lambda_gcb_psi', 1.0)) * l_psi
                 + float(tc.get('lambda_gcb_reward', 1.0)) * l_rew)
        terms.update(gcb_phi=l_phi, gcb_psi=l_psi, gcb_reward=l_rew)

    terms['total'] = total
    if with_diagnostics:  # both prediction errors for every model, comparable across objectives
        with torch.no_grad():
            other = 'multi_step' if tc['prediction'] == 'one_step' else 'one_step'
            l_other, _ = prediction_loss(model, z, actions, other, ctx)
            terms[f'err_{tc["prediction"]}'] = l_pred.detach()
            terms[f'err_{other}'] = l_other
            # MSE per element / variance per dimension: 1.0 = no better than predicting the mean
            var = z.reshape(-1, z.shape[-1]).var(0).mean()
            terms['rel_err_multi_step'] = terms['err_multi_step'] / var.clamp_min(1e-12)
            terms['latent_std'] = z.reshape(-1, z.shape[-1]).std(0).mean()
    return total, {k: float(v.detach()) for k, v in terms.items()}, z
