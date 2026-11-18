"""Diffusion Policy (Chi et al., RSS 2023), CNN variant, for the cached Franka dataset.

Observation (2 steps, t - obs_skip and t):
    images  ResNet-18 with GroupNorm (no pretrained weights needed), random crop in training
    proprio the same normalized [q, dq, gripper] vector as the world models
    task    learned embedding of the task id (pick vs rotate of one object look alike at the start)
Action: the next `horizon` per-step actions [dpos (3), drot (3), gripper command (1)], the same
    'delta_next' quantity the world models are trained on, min-max normalized to [-1, 1].
Denoiser: 1-D temporal U-Net with FiLM conditioning; DDPM (100 steps, squared-cosine schedule,
    epsilon prediction) for training, DDIM for sampling.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa.data import proprio_array


# --------------------------------------------------------------------------- #
# Vision: ResNet-18 with GroupNorm
# --------------------------------------------------------------------------- #

def _gn(c):
    return nn.GroupNorm(max(1, c // 16), c)


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = _gn(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = _gn(cout)
        self.skip = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), _gn(cout))
                     if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        y = F.relu(self.n1(self.c1(x)))
        return F.relu(self.n2(self.c2(y)) + self.skip(x))


class ResNet18GN(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), _gn(64), nn.ReLU(),
                                  nn.MaxPool2d(3, 2, 1))
        layers, cin = [], 64
        for cout, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            layers += [BasicBlock(cin, cout, stride), BasicBlock(cout, cout)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.out = nn.Linear(512, out_dim)

    def forward(self, x):
        return self.out(self.layers(self.stem(x)).mean((2, 3)))


# --------------------------------------------------------------------------- #
# Temporal U-Net (as in the Diffusion Policy reference implementation)
# --------------------------------------------------------------------------- #

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000) / (half - 1)))
        e = t.float()[:, None] * f[None]
        return torch.cat([e.sin(), e.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, cin, cout, k=5, groups=8):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.GroupNorm(groups, cout), nn.Mish())

    def forward(self, x):
        return self.block(x)


class CondResBlock1D(nn.Module):
    def __init__(self, cin, cout, cond_dim, k=5, groups=8):
        super().__init__()
        self.b1 = Conv1dBlock(cin, cout, k, groups)
        self.b2 = Conv1dBlock(cout, cout, k, groups)
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * cout))
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        y = self.b1(x)
        scale, bias = self.film(cond)[:, :, None].chunk(2, dim=1)
        y = self.b2(scale * y + bias)
        return y + self.res(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, action_dim, cond_dim, down_dims=(128, 256, 512), step_dim=128, k=5, groups=8):
        super().__init__()
        self.step_emb = nn.Sequential(SinusoidalPosEmb(step_dim), nn.Linear(step_dim, 4 * step_dim), nn.Mish(),
                                      nn.Linear(4 * step_dim, step_dim))
        cd = step_dim + cond_dim
        dims = [action_dim] + list(down_dims)
        pairs = list(zip(dims[:-1], dims[1:]))
        self.down = nn.ModuleList()
        for i, (a, b) in enumerate(pairs):
            last = i == len(pairs) - 1
            self.down.append(nn.ModuleList([CondResBlock1D(a, b, cd, k, groups), CondResBlock1D(b, b, cd, k, groups),
                                            nn.Conv1d(b, b, 3, 2, 1) if not last else nn.Identity()]))
        m = dims[-1]
        self.mid = nn.ModuleList([CondResBlock1D(m, m, cd, k, groups), CondResBlock1D(m, m, cd, k, groups)])
        self.up = nn.ModuleList()
        for a, b in reversed(pairs[1:]):  # one upsampling per downsampling (as in the reference code)
            self.up.append(nn.ModuleList([CondResBlock1D(2 * b, a, cd, k, groups), CondResBlock1D(a, a, cd, k, groups),
                                          nn.ConvTranspose1d(a, a, 4, 2, 1)]))
        s = down_dims[0]
        self.final = nn.Sequential(Conv1dBlock(s, s, k, groups), nn.Conv1d(s, action_dim, 1))

    def forward(self, x, t, cond):
        """x: (B, T, A) noisy actions; t: (B,) diffusion step; cond: (B, C) -> (B, T, A)."""
        c = torch.cat([self.step_emb(t), cond], dim=-1)
        x = x.transpose(1, 2)
        skips = []
        for r1, r2, down in self.down:
            x = r2(r1(x, c), c)
            skips.append(x)
            x = down(x)
        for r in self.mid:
            x = r(x, c)
        for r1, r2, up in self.up:
            x = torch.cat([x, skips.pop()], dim=1)
            x = up(r2(r1(x, c), c))
        return self.final(x).transpose(1, 2)


# --------------------------------------------------------------------------- #
# Noise schedule
# --------------------------------------------------------------------------- #

def squared_cosine_alphas_cumprod(n, s=0.008):
    f = lambda t: math.cos((t / n + s) / (1 + s) * math.pi / 2) ** 2
    betas = [min(1 - f(i + 1) / f(i), 0.999) for i in range(n)]
    return torch.cumprod(1 - torch.tensor(betas, dtype=torch.float64), 0).float()


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #

class DiffusionPolicy(nn.Module):
    def __init__(self, n_tasks, proprio_dim, action_dim=7, horizon=16, n_obs=2, image_size=128,
                 crop=0.9, vision_dim=512, task_dim=32, down_dims=(128, 256, 512), train_steps=100,
                 goal_cond=False):
        super().__init__()
        self.cfg = dict(n_tasks=n_tasks, proprio_dim=proprio_dim, action_dim=action_dim, horizon=horizon,
                        n_obs=n_obs, image_size=image_size, crop=crop, vision_dim=vision_dim,
                        task_dim=task_dim, down_dims=list(down_dims), train_steps=train_steps,
                        goal_cond=bool(goal_cond))
        self.goal_cond = bool(goal_cond)
        self.horizon, self.action_dim, self.n_obs = horizon, action_dim, n_obs
        self.crop_size = int(round(crop * image_size))
        self.vision = ResNet18GN(vision_dim)
        self.task_emb = nn.Embedding(n_tasks, task_dim)
        cond_dim = n_obs * (vision_dim + proprio_dim) + task_dim + (vision_dim if goal_cond else 0)
        self.unet = ConditionalUnet1D(action_dim, cond_dim, tuple(down_dims))
        self.train_steps = train_steps
        self.register_buffer('alphas_cumprod', squared_cosine_alphas_cumprod(train_steps))

    def encode_obs(self, images, proprio, task, goal=None):
        """images: (B, n_obs, 3, S, S) uint8; proprio: (B, n_obs, P) normalized; task: (B,);
        goal: (B, 3, S, S) uint8 goal image (goal-conditioned policies) -> (B, C)."""
        if self.goal_cond:
            if goal is None:
                raise ValueError('this policy is goal-conditioned: pass the goal image')
            images = torch.cat([images, goal[:, None]], dim=1)
        x = images.float() / 255.0
        x = self._crop_batch(x)
        b, n = x.shape[:2]
        f = self.vision((x.reshape(b * n, *x.shape[2:]) - 0.5) / 0.25).reshape(b, n, -1)
        return torch.cat([f.reshape(b, -1), proprio.reshape(b, -1), self.task_emb(task)], dim=-1)

    def _crop_batch(self, x):
        s, c = x.shape[-1], self.crop_size
        if c >= s:
            return x
        if not self.training:
            o = (s - c) // 2
            return x[..., o:o + c, o:o + c]
        b = x.shape[0]
        i = torch.randint(0, s - c + 1, (b,)).tolist()
        j = torch.randint(0, s - c + 1, (b,)).tolist()
        return torch.stack([x[k, ..., i[k]:i[k] + c, j[k]:j[k] + c] for k in range(b)])

    def loss(self, images, proprio, task, actions, goal=None):
        """actions: (B, T, A) normalized to [-1, 1]. Epsilon-prediction DDPM loss."""
        cond = self.encode_obs(images, proprio, task, goal)
        b = actions.shape[0]
        t = torch.randint(0, self.train_steps, (b,), device=actions.device)
        ac = self.alphas_cumprod[t][:, None, None]
        eps = torch.randn_like(actions)
        noisy = ac.sqrt() * actions + (1 - ac).sqrt() * eps
        return F.mse_loss(self.unet(noisy, t, cond), eps)

    @torch.no_grad()
    def sample(self, cond, n_steps=16, generator=None):
        """DDIM (eta = 0) from cond (B, C) -> (B, T, A) normalized actions in [-1, 1]."""
        b = cond.shape[0]
        x = torch.randn(b, self.horizon, self.action_dim, device=cond.device, generator=generator)
        ratio = self.train_steps // n_steps
        ts = (np.arange(n_steps) * ratio)[::-1]
        for i, t in enumerate(ts):
            tt = torch.full((b,), int(t), device=cond.device, dtype=torch.long)
            eps = self.unet(x, tt, cond)
            a_t = self.alphas_cumprod[int(t)]
            a_prev = self.alphas_cumprod[int(ts[i + 1])] if i + 1 < len(ts) else torch.tensor(1.0, device=cond.device)
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
            eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x.clamp(-1, 1)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

class PolicyData:
    """Observation / action-chunk access on the cache, with the normalizations of training."""

    def __init__(self, cache, horizon=16, obs_skip=2, proprio_keys=('q', 'dq', 'gripper'), stats=None,
                 train_episodes=None, action_mode='delta_next', goal_p=0.01):
        self.c, self.horizon, self.obs_skip = cache, horizon, obs_skip
        self.action_mode, self.goal_p = action_mode, float(goal_p)
        self.actions = cache.actions(action_mode)
        self.proprio_raw = proprio_array(cache, list(proprio_keys))
        if stats is None:
            rows = np.concatenate([np.arange(cache.starts[e], cache.starts[e] + cache.lengths[e] - 1)
                                   for e in train_episodes])
            a, p = self.actions[rows], self.proprio_raw[np.concatenate(
                [np.arange(cache.starts[e], cache.starts[e] + cache.lengths[e]) for e in train_episodes])]
            stats = {'a_min': a.min(0).tolist(), 'a_max': a.max(0).tolist(), 'p_mean': p.mean(0).tolist(),
                     'p_std': np.maximum(p.std(0), 1e-6).tolist(), 'proprio_keys': list(proprio_keys)}
        self.stats = stats
        self.a_min, self.a_max = np.array(stats['a_min'], np.float32), np.array(stats['a_max'], np.float32)
        self.proprio = ((self.proprio_raw - np.array(stats['p_mean'], np.float32))
                        / np.array(stats['p_std'], np.float32)).astype(np.float32)
        self.ep_start = cache.starts[cache.episode_of_frame]
        self.ep_end = self.ep_start + cache.lengths[cache.episode_of_frame] - 1   # last frame index

    def normalize(self, a):
        rng = np.maximum(self.a_max - self.a_min, 1e-8)
        return np.where(self.a_max - self.a_min > 1e-8, 2 * (a - self.a_min) / rng - 1, 0.0).astype(np.float32)

    def unnormalize(self, a):
        return ((a + 1) / 2 * (self.a_max - self.a_min) + self.a_min).astype(np.float32)

    def obs_frames(self, t):
        """(n, 2) frame indices [t - obs_skip, t], clipped to the episode start."""
        t = np.asarray(t)
        return np.stack([np.maximum(t - self.obs_skip, self.ep_start[t]), t], axis=1)

    def action_rows(self, t):
        """(n, horizon) frame indices of the action chunk from t; past the episode end the last
        row (zero motion, last gripper command) is repeated."""
        t = np.asarray(t)
        return np.minimum(t[:, None] + np.arange(self.horizon)[None], self.ep_end[t][:, None])

    def sample_goal_frames(self, t, rng):
        """Hindsight goals for goal-conditioned training: a later frame of the same episode,
        k ~ Geometric(goal_p) frames ahead (mean 1/goal_p), clipped to the episode end."""
        t = np.asarray(t)
        return np.minimum(t + rng.geometric(self.goal_p, size=t.shape), self.ep_end[t])

    def goal_images(self, g):
        return torch.from_numpy(np.ascontiguousarray(self.c.images[np.asarray(g)])).permute(0, 3, 1, 2)

    def batch(self, t, device):
        f = self.obs_frames(t)
        img = torch.from_numpy(np.ascontiguousarray(self.c.images[f.reshape(-1)])).reshape(
            len(t), 2, *self.c.images.shape[1:]).permute(0, 1, 4, 2, 3)
        prop = torch.from_numpy(self.proprio[f])
        task = torch.from_numpy(self.c.episode_task[self.c.episode_of_frame[t]].astype(np.int64))
        act = torch.from_numpy(self.normalize(self.actions[self.action_rows(t)]))
        return img.to(device), prop.to(device), task.to(device), act.to(device)


def load_policy(path, device):
    from jepa.models import torch_load
    ck = torch_load(path, device)
    cfg = dict(ck['config'])
    cfg['down_dims'] = tuple(cfg['down_dims'])
    pol = DiffusionPolicy(**{k: v for k, v in cfg.items()}).to(device)
    pol.load_state_dict(ck['ema'] if 'ema' in ck else ck['model'])
    pol.eval()
    return pol, ck
