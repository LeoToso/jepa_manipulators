"""JEPA world model components.

Encoder, action embedder, AdaLN-zero causal predictor and endpoint inverse
model are adapted from JEPA_control (models/jepa_world_model.py, Toso et al.),
whose architecture follows the Sensorimotor World Model / LeWorldModel. The
goal-conditioned bisimulation heads follow Hansen-Estruch et al., ICML 2022.

Changes relative to JEPA_control:
- the predictor is used strictly as a Markov one-step model f(z_t, a_t), both
  for one-step (teacher-forced) and multi-step (autoregressive) losses, so the
  two objectives train the same function that planning uses;
- attention falls back to an explicit implementation on torch < 2.0;
- the encoder can return patch tokens (for patch-PCA visualizations).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_attention(q, k, v, dropout_p):
    if hasattr(F, 'scaled_dot_product_attention'):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
    t = q.shape[-2]
    att = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    mask = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    att = att.masked_fill(~mask, float('-inf')).softmax(-1)
    return F.dropout(att, dropout_p) @ v


class _EncoderBlock(nn.Module):
    """Pre-LN transformer block (same as nn.TransformerEncoderLayer(norm_first=True, gelu))."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class PatchViTEncoder(nn.Module):
    """ViT-Tiny-style encoder: patches -> transformer -> CLS -> MLP projector (BN)."""

    def __init__(self, image_size=128, patch_size=8, in_channels=3, embed_dim=192, depth=4,
                 heads=3, mlp_ratio=4.0, latent_dim=192):
        super().__init__()
        if image_size % patch_size:
            raise ValueError('image_size must be divisible by patch_size')
        self.grid = image_size // patch_size
        self.patch = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, self.grid ** 2 + 1, embed_dim))
        self.blocks = nn.ModuleList([_EncoderBlock(embed_dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.projector = nn.Sequential(nn.Linear(embed_dim, 2048), nn.BatchNorm1d(2048), nn.GELU(),
                                       nn.Linear(2048, latent_dim))
        nn.init.trunc_normal_(self.cls, std=.02)
        nn.init.trunc_normal_(self.pos, std=.02)

    def forward(self, pixels, return_tokens=False):
        x = self.patch(pixels).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls.expand(x.shape[0], -1, -1), x), dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        z = self.projector(x[:, 0])
        return (z, x[:, 1:]) if return_tokens else z


class ActionEmbedder(nn.Module):
    def __init__(self, action_dim, embed_dim, mlp_scale=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(action_dim, mlp_scale * embed_dim), nn.SiLU(),
                                 nn.Linear(mlp_scale * embed_dim, embed_dim))

    def forward(self, action):
        return self.net(action.float())


def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ConditionalBlock(nn.Module):
    """Causal transformer block with AdaLN-zero action conditioning."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.heads, self.dim_head = heads, dim_head
        inner = heads * dim_head
        self.to_qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.to_out = nn.Linear(inner, dim)
        self.dropout = dropout
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, dim), nn.Dropout(dropout))
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x, c):
        s1, k1, g1, s2, k2, g2 = self.adaln(c).chunk(6, dim=-1)
        h = _modulate(self.norm1(x), s1, k1)
        b, t, _ = h.shape
        q, k, v = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = (y.reshape(b, t, self.heads, self.dim_head).transpose(1, 2) for y in (q, k, v))
        h = _causal_attention(q, k, v, self.dropout if self.training else 0.)
        x = x + g1 * self.to_out(h.transpose(1, 2).reshape(b, t, -1))
        return x + g2 * self.mlp(_modulate(self.norm2(x), s2, k2))


class ARPredictor(nn.Module):
    def __init__(self, latent_dim=192, hidden_dim=192, depth=6, heads=16, dim_head=64,
                 mlp_dim=2048, dropout=.1):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_dim) if latent_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(latent_dim, hidden_dim) if latent_dim != hidden_dim else nn.Identity()
        self.pos = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.blocks = nn.ModuleList([ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, latent_dim) if hidden_dim != latent_dim else nn.Identity()
        self.pred_projector = nn.Sequential(nn.Linear(latent_dim, 2048), nn.BatchNorm1d(2048),
                                            nn.GELU(), nn.Linear(2048, latent_dim))

    def forward(self, z, action_embedding):
        """z, action_embedding: (b, t, d) -> (b, t, d)."""
        x = self.input_proj(z + self.pos)
        c = self.cond_proj(action_embedding)
        for block in self.blocks:
            x = block(x, c)
        x = self.output(self.norm(x))
        b, t, d = x.shape
        return self.pred_projector(x.reshape(b * t, d)).reshape(b, t, d)


class EndpointInverseModel(nn.Module):
    """EP-IDM: the H-step action sequence from the endpoints (z_t, z_{t+H}) only."""

    def __init__(self, latent_dim=192, action_dim=14, hidden_dim=256, horizon=3):
        super().__init__()
        self.horizon, self.action_dim = int(horizon), int(action_dim)
        self.net = nn.Sequential(nn.Linear(2 * latent_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, self.horizon * self.action_dim))

    def forward(self, z_start, z_end):
        x = self.net(torch.cat([z_start, z_end], dim=-1))
        return x.reshape(x.shape[0], self.horizon, self.action_dim)


class GCBHeads(nn.Module):
    """Goal-conditioned bisimulation heads on top of the JEPA latent psi(s) = z.

    phi(s, g): paired state-goal embedding, an MLP of [z_s, z_g, z_g - z_s] with the
               same dimension as z so that the analogy psi(g) - psi(s) = phi(s, g)
               (Hansen-Estruch et al., Eq. 5/7) is well defined.
    reward:    R(phi(s, g), phi(s', g)) -> probability that s' is the goal (their D.5).
    """

    def __init__(self, latent_dim=192, hidden_dim=512):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(3 * latent_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                 nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, latent_dim))
        self.reward = nn.Sequential(nn.Linear(2 * latent_dim, hidden_dim // 2), nn.GELU(),
                                    nn.Linear(hidden_dim // 2, 1))

    def pair(self, z_s, z_g):
        return self.phi(torch.cat([z_s, z_g, z_g - z_s], dim=-1))


class WorldModel(nn.Module):
    """encode(obs, prev) -> z;  step(z, a) -> z';  rollout(z0, a_1..H) -> z_1..H."""

    def __init__(self, cfg, action_dim):
        super().__init__()
        self.cfg = dict(cfg)
        d = int(cfg.get('latent_dim', 192))
        self.latent_dim, self.action_dim = d, int(action_dim)
        self.use_frame_diff = bool(cfg.get('use_frame_diff', True))
        self.encoder = PatchViTEncoder(
            image_size=int(cfg.get('image_size', 128)), patch_size=int(cfg.get('patch_size', 8)),
            in_channels=9 if self.use_frame_diff else 3, embed_dim=int(cfg.get('vit_embed_dim', 192)),
            depth=int(cfg.get('vit_depth', 4)), heads=int(cfg.get('vit_heads', 3)), latent_dim=d)
        self.action_encoder = ActionEmbedder(self.action_dim, d)
        self.predictor = ARPredictor(
            latent_dim=d, hidden_dim=int(cfg.get('predictor_hidden_dim', 192)),
            depth=int(cfg.get('predictor_depth', 6)), heads=int(cfg.get('predictor_heads', 16)),
            dim_head=int(cfg.get('predictor_dim_head', 64)),
            mlp_dim=int(cfg.get('predictor_mlp_dim', 2048)),
            dropout=float(cfg.get('predictor_dropout', .1)))
        self.endpoint_inverse = (EndpointInverseModel(
            d, self.action_dim, int(cfg.get('inverse_hidden_dim', 256)),
            int(cfg.get('endpoint_inverse_horizon', 3)))
            if cfg.get('use_endpoint_inverse', False) else None)
        self.gcb = (GCBHeads(d, int(cfg.get('gcb_hidden_dim', 512)))
                    if cfg.get('use_gcb', False) else None)

    @staticmethod
    def to_float(x):
        return x.float().div(255.) if x.dtype == torch.uint8 else x.float()

    def encode(self, obs, prev_obs=None, return_tokens=False):
        """obs, prev_obs: (B, 3, S, S) uint8 or float in [0, 1]."""
        obs = self.to_float(obs)
        if self.use_frame_diff:
            prev = obs if prev_obs is None else self.to_float(prev_obs)
            obs = torch.cat((prev, obs, obs - prev), dim=1)
        return self.encoder(obs, return_tokens=return_tokens)

    def encode_seq(self, obs, prev_obs=None):
        """(B, T, 3, S, S) -> (B, T, d)."""
        b, t = obs.shape[:2]
        flat = lambda x: None if x is None else x.reshape(b * t, *x.shape[2:])
        return self.encode(flat(obs), flat(prev_obs)).reshape(b, t, -1)

    def step(self, z, a):
        """Markov one-step prediction: (B, d), (B, A) -> (B, d)."""
        return self.predictor(z[:, None], self.action_encoder(a)[:, None])[:, 0]

    def rollout(self, z0, actions):
        """Autoregressive: (B, d), (B, H, A) -> (B, H, d), gradients through the chain."""
        z, out = z0, []
        for k in range(actions.shape[1]):
            z = self.step(z, actions[:, k])
            out.append(z)
        return torch.stack(out, dim=1)


def torch_load(path, device='cpu'):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only
        return torch.load(path, map_location=device)


def load_checkpoint(path, device='cpu'):
    ckpt = torch_load(path, device)
    model = WorldModel(ckpt['config']['model'], ckpt['action_dim']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt
