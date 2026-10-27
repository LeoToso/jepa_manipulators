#!/usr/bin/env python3
"""PyTorch dataset for the Franka FR3 recordings (Dataset/*.h5).

The raw files store camera frames gzip-compressed at 376x672, which is too
slow to decode at every training step. So there are two stages:

1. Build a cache once (resizes the images and stores everything uncompressed):

       python franka_dataset.py build-cache Dataset --out cache_224 --size 224 224

   Use `--crop center` to crop the 672x376 frames to a square before resizing
   instead of stretching them.

2. Train from the cache:

       from franka_dataset import FrankaDataset, split_episodes
       train_eps, val_eps = split_episodes("cache_224", val_fraction=0.15)
       train = FrankaDataset("cache_224", episodes=train_eps, action_mode="delta")
       val = FrankaDataset("cache_224", episodes=val_eps, action_mode="delta",
                           stats=train.stats)
       loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True,
                                            num_workers=4)

Each sample is a dict:
    image   float32 [0, 1], (3, H, W), or (T, 3, H, W) when num_frames > 1
    state   float32 (D,)  or (T, D)   default: q (7) + x_ee (7) + gripper (1) = 15
    action  float32 (A,)  or (T, A)
    episode, step   int  where the (first) frame comes from

Action modes (the gripper command is always kept as is, the last dimension):
    "absolute"    x_ee_des: target position (3) + quaternion (4) + gripper = 8 dims
    "delta"       target minus current pose: dpos (3) + rotation vector (3) + gripper = 7
    "delta_next"  next pose minus current pose: dpos (3) + rotvec (3) + gripper = 7
                  (the last step of each episode has no next pose and is skipped)

The rotation deltas need the order of the stored quaternions. Check it with

    python franka_dataset.py check-quat Dataset

and pass the result as quat_order= ("wxyz" or "xyzw").

Other commands:
    python franka_dataset.py info cache_224      # summary of a cache
    python franka_dataset.py demo cache_224      # load a batch, print shapes
"""

import argparse
import glob
import json
import os

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # building the cache does not need torch
    torch = None
    _TorchDataset = object

LOWDIM_KEYS = ["fr3/q", "fr3/dq", "fr3/x_ee", "fr3/v_ee", "fr3/gripper",
               "action/x_ee_des", "action/gripper", "timestamps"]
DEFAULT_STATE_KEYS = ("fr3/q", "fr3/x_ee", "fr3/gripper")


# --------------------------------------------------------------------------- #
# Quaternion helpers (numpy, (..., 4) arrays in w, x, y, z order)
# --------------------------------------------------------------------------- #

def to_wxyz(quat, order):
    if order == "wxyz":
        return quat
    if order == "xyzw":
        return np.roll(quat, 1, axis=-1)
    raise ValueError(f"quat_order must be 'wxyz' or 'xyzw', got {order!r}")


def quat_conj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)


def quat_mul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def quat_to_rotvec(q):
    """Rotation vector (axis * angle, radians) of unit quaternions."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[..., :1] < 0, -q, q)  # shortest rotation
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, q[..., :1])
    # angle / sin(angle / 2) -> 2 as s -> 0
    scale = np.where(s > 1e-8, angle / np.maximum(s, 1e-8), 2.0)
    return v * scale


def relative_rotvec(q_from, q_to):
    """World-frame rotation vector taking orientation q_from to q_to."""
    return quat_to_rotvec(quat_mul(q_to, quat_conj(q_from)))


# --------------------------------------------------------------------------- #
# Reading the raw .h5 files
# --------------------------------------------------------------------------- #

def find_h5_files(path):
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "*.h5")) + glob.glob(os.path.join(path, "*.hdf5")))
    if not files:
        raise SystemExit(f"no .h5 files found in {path}")
    return files


def iter_episodes(path):
    """Yield (file, episode_name, h5 group) for every episode, in order."""
    import h5py
    for fname in find_h5_files(path):
        with h5py.File(fname, "r") as f:
            for name in sorted(f.keys()):
                if isinstance(f[name], h5py.Group):
                    yield fname, name, f[name]


def resize_frames(frames, size, crop):
    from PIL import Image
    h, w = size
    out = np.empty((len(frames), h, w, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        if crop == "center":
            fh, fw = frame.shape[:2]
            side = min(fh, fw)
            top, left = (fh - side) // 2, (fw - side) // 2
            frame = frame[top:top + side, left:left + side]
        out[i] = np.asarray(Image.fromarray(frame).resize((w, h), Image.BILINEAR))
    return out


# --------------------------------------------------------------------------- #
# Cache building
# --------------------------------------------------------------------------- #

def build_cache(src, out, size=(224, 224), crop="none", camera=None, chunk=64):
    """Write resized images (images.npy, memory-mappable) + low-dim arrays."""
    episodes = []
    total = 0
    for fname, name, ep in iter_episodes(src):
        cams = sorted(ep["cams"].keys())
        cam = camera or cams[0]
        episodes.append((fname, name, cam, ep["timestamps"].shape[0]))
        total += episodes[-1][3]
    print(f"{len(episodes)} episodes, {total} steps -> {out}  (images {size[0]}x{size[1]}, "
          f"crop={crop}, {total * size[0] * size[1] * 3 / 1e9:.2f} GB)")

    os.makedirs(out, exist_ok=True)
    images = np.lib.format.open_memmap(os.path.join(out, "images.npy"), mode="w+",
                                       dtype=np.uint8, shape=(total, size[0], size[1], 3))
    lowdim = {k: [] for k in LOWDIM_KEYS}

    import h5py
    pos = 0
    for i, (fname, name, cam, n) in enumerate(episodes):
        with h5py.File(fname, "r") as f:
            ep = f[name]
            rgb = ep["cams"][cam]["rgb"]
            for start in range(0, n, chunk):
                frames = rgb[start:start + chunk]
                images[pos + start:pos + start + len(frames)] = resize_frames(frames, size, crop)
            for k in LOWDIM_KEYS:
                arr = ep[k][()]
                lowdim[k].append(arr.reshape(n, -1) if k != "timestamps" else arr)
        pos += n
        print(f"  [{i + 1}/{len(episodes)}] {os.path.basename(fname)} {name}: {n} steps")
    images.flush()

    lengths = np.array([e[3] for e in episodes])
    np.savez(os.path.join(out, "lowdim.npz"),
             episode_lengths=lengths,
             episode_starts=np.concatenate([[0], np.cumsum(lengths)[:-1]]),
             **{k: np.concatenate(v).astype(np.float64 if k == "timestamps" else np.float32)
                for k, v in lowdim.items()})
    info = {
        "source": os.path.abspath(src),
        "image_size": list(size),
        "crop": crop,
        "num_steps": int(total),
        "episodes": [{"file": os.path.basename(e[0]), "name": e[1], "camera": e[2],
                      "length": int(e[3])} for e in episodes],
    }
    with open(os.path.join(out, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"done: {out}")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def split_episodes(cache_dir, val_fraction=0.15, seed=0):
    """Random train/val split by whole episodes (never mix steps of one episode)."""
    n = len(np.load(os.path.join(cache_dir, "lowdim.npz"))["episode_lengths"])
    perm = np.random.RandomState(seed).permutation(n)
    n_val = max(1, int(round(n * val_fraction))) if n > 1 else 0
    return sorted(perm[n_val:].tolist()), sorted(perm[:n_val].tolist())


def compute_actions(low, mode, quat_order):
    """Per-step action array for the whole cache, shape (N, A)."""
    x_ee, x_des, grip = low["fr3/x_ee"], low["action/x_ee_des"], low["action/gripper"]
    if mode == "absolute":
        return np.concatenate([x_des, grip], axis=1)

    q_cur = to_wxyz(x_ee[:, 3:7].astype(np.float64), quat_order)
    if mode == "delta":
        target_pos, q_tgt = x_des[:, :3], to_wxyz(x_des[:, 3:7].astype(np.float64), quat_order)
    elif mode == "delta_next":
        # next step's pose; the last step of an episode is never used (see valid starts)
        target_pos = np.concatenate([x_ee[1:, :3], x_ee[-1:, :3]])
        q_tgt = np.concatenate([q_cur[1:], q_cur[-1:]])
    else:
        raise ValueError(f"unknown action_mode {mode!r}")
    dpos = target_pos - x_ee[:, :3]
    drot = relative_rotvec(q_cur, q_tgt)
    return np.concatenate([dpos, drot, grip], axis=1).astype(np.float32)


class FrankaDataset(_TorchDataset):
    """Samples (image, state, action) windows from a cache made by build_cache.

    Args:
        cache_dir: folder written by `build-cache`.
        episodes: episode indices to use (default: all). See split_episodes.
        num_frames: consecutive steps per sample (1 = single step, no time dim).
        frame_skip: stride between those steps (e.g. 2 -> 5 Hz from 10 Hz data).
        action_mode: "absolute", "delta" or "delta_next" (see module docstring).
        quat_order: "wxyz" or "xyzw"; run `check-quat` on the raw data.
        state_keys: low-dim arrays concatenated into the state vector.
        normalize: standardize state/action with mean/std (gripper dims included).
        stats: mean/std dict to reuse, e.g. pass train.stats to the val set.
        transform: optional callable applied to the image tensor.
    """

    def __init__(self, cache_dir, episodes=None, num_frames=1, frame_skip=1,
                 action_mode="absolute", quat_order="wxyz", state_keys=DEFAULT_STATE_KEYS,
                 normalize=False, stats=None, transform=None):
        if torch is None:
            raise ImportError("FrankaDataset needs PyTorch")
        self.cache_dir = cache_dir
        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.transform = transform
        self.normalize = normalize
        self._images = None  # opened lazily so every DataLoader worker maps it itself

        low = dict(np.load(os.path.join(cache_dir, "lowdim.npz")))
        lengths, starts = low["episode_lengths"], low["episode_starts"]
        if episodes is None:
            episodes = list(range(len(lengths)))
        self.episodes = list(episodes)

        self.state = np.concatenate([low[k] for k in state_keys], axis=1).astype(np.float32)
        self.action = compute_actions(low, action_mode, quat_order)
        self.episode_of_step = np.repeat(np.arange(len(lengths)), lengths)
        self.step_in_episode = np.concatenate([np.arange(n) for n in lengths])

        # delta_next has no action at the last step of an episode
        drop_last = 1 if action_mode == "delta_next" else 0
        span = (num_frames - 1) * frame_skip + 1 + drop_last
        windows = [np.arange(starts[e], starts[e] + lengths[e] - span + 1)
                   for e in self.episodes if lengths[e] >= span]
        if not windows:
            raise ValueError("no samples: episodes too short for num_frames/frame_skip")
        self.index = np.concatenate(windows).astype(np.int64)

        if stats is None:
            used = np.concatenate([np.arange(starts[e], starts[e] + lengths[e] - drop_last)
                                   for e in self.episodes])
            stats = {
                "state_mean": self.state[used].mean(0), "state_std": self.state[used].std(0) + 1e-6,
                "action_mean": self.action[used].mean(0), "action_std": self.action[used].std(0) + 1e-6,
            }
        self.stats = stats
        if normalize:
            self.state = (self.state - stats["state_mean"]) / stats["state_std"]
            self.action = (self.action - stats["action_mean"]) / stats["action_std"]

    @property
    def state_dim(self):
        return self.state.shape[1]

    @property
    def action_dim(self):
        return self.action.shape[1]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        if self._images is None:
            self._images = np.load(os.path.join(self.cache_dir, "images.npy"), mmap_mode="r")
        start = self.index[i]
        steps = start + np.arange(self.num_frames) * self.frame_skip

        image = torch.from_numpy(np.ascontiguousarray(self._images[steps]))
        image = image.permute(0, 3, 1, 2).float().div_(255.0)  # (T, 3, H, W)
        state = torch.from_numpy(self.state[steps])
        action = torch.from_numpy(self.action[steps])
        if self.num_frames == 1:
            image, state, action = image[0], state[0], action[0]
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "state": state, "action": action,
                "episode": int(self.episode_of_step[start]),
                "step": int(self.step_in_episode[start])}


# --------------------------------------------------------------------------- #
# Checks / command line
# --------------------------------------------------------------------------- #

def check_quat_order(src):
    """Find the quaternion order by comparing finite-differenced orientations
    with the measured angular velocity in fr3/v_ee[:, 3:6]."""
    results = {}
    lin = []
    for _, _, ep in iter_episodes(src):
        x, v, t = ep["fr3/x_ee"][()].astype(np.float64), ep["fr3/v_ee"][()].astype(np.float64), ep["timestamps"][()]
        dt = np.diff(t)[:, None]
        v_mid = 0.5 * (v[1:] + v[:-1])  # velocity halfway between two samples
        lin.append(_cosine((x[1:, :3] - x[:-1, :3]) / dt, v_mid[:, :3]))
        for order in ("wxyz", "xyzw"):
            q = to_wxyz(x[:, 3:7], order)
            for frame in ("base", "end-effector"):
                if frame == "base":
                    rel = quat_mul(q[1:], quat_conj(q[:-1]))
                else:
                    rel = quat_mul(quat_conj(q[:-1]), q[1:])
                omega = quat_to_rotvec(rel) / dt
                results.setdefault((order, frame), []).append(_cosine(omega, v_mid[:, 3:6]))

    print("Agreement between pose changes and measured velocity v_ee "
          "(cosine similarity, 1.0 = perfect):")
    print(f"  linear velocity  vs position change:  {np.nanmean(lin):.3f}")
    best = None
    for (order, frame), vals in results.items():
        score = np.nanmean(vals)
        print(f"  angular velocity, quat_order={order}, {frame:12s} frame:  {score:.3f}")
        if best is None or score > best[1]:
            best = ((order, frame), score)
    (order, frame), score = best
    if score > 0.8:
        print(f"\n-> quaternions are stored as {order} (angular velocity in {frame} frame). "
              f"Use quat_order=\"{order}\".")
    else:
        print("\n-> no convention matches well; the rotation may be too small to tell, or "
              "v_ee is not [linear, angular]. Check the logging code.")
    return order


def _cosine(a, b, min_norm=0.02):
    """Mean cosine similarity over steps where the measured vector b is not ~0."""
    nb = np.linalg.norm(b, axis=1)
    na = np.linalg.norm(a, axis=1)
    keep = (nb > min_norm) & (na > 1e-9)
    if not keep.any():
        return np.nan
    return float(np.mean(np.sum(a[keep] * b[keep], axis=1) / (na[keep] * nb[keep])))


def print_info(cache_dir):
    with open(os.path.join(cache_dir, "info.json")) as f:
        info = json.load(f)
    print(f"cache {cache_dir}: {info['num_steps']} steps, images {info['image_size']}, "
          f"crop={info['crop']}, source={info['source']}")
    for i, e in enumerate(info["episodes"]):
        print(f"  {i:3d}  {e['file']}  {e['name']}  {e['length']} steps")


def demo(cache_dir, quat_order):
    from torch.utils.data import DataLoader
    train_eps, val_eps = split_episodes(cache_dir)
    print(f"train episodes {train_eps}, val episodes {val_eps}")
    for mode in ("absolute", "delta", "delta_next"):
        ds = FrankaDataset(cache_dir, episodes=train_eps, action_mode=mode, quat_order=quat_order)
        a = ds.action[ds.index]  # only steps that samples start from
        print(f"\naction_mode={mode}: {len(ds)} samples, state_dim={ds.state_dim}, "
              f"action_dim={ds.action_dim}")
        print("  action mean:", np.array2string(a.mean(0), precision=4, suppress_small=True))
        print("  action std: ", np.array2string(a.std(0), precision=4, suppress_small=True))
    ds = FrankaDataset(cache_dir, episodes=train_eps, num_frames=4, frame_skip=2,
                       action_mode="delta", quat_order=quat_order, normalize=True)
    batch = next(iter(DataLoader(ds, batch_size=8, shuffle=True)))
    print("\nbatch with num_frames=4, frame_skip=2, normalize=True:")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-cache", help="resize images and write a training cache")
    p.add_argument("src", help="Dataset folder or a single .h5 file")
    p.add_argument("--out", required=True, help="output cache folder")
    p.add_argument("--size", type=int, nargs=2, default=[224, 224], metavar=("H", "W"))
    p.add_argument("--crop", choices=["none", "center"], default="none",
                   help="'center' crops to a square first; 'none' stretches (default)")
    p.add_argument("--camera", default=None, help="camera serial (default: first)")

    p = sub.add_parser("check-quat", help="detect the quaternion order in the raw data")
    p.add_argument("src")

    p = sub.add_parser("info", help="summarize a cache")
    p.add_argument("cache")

    p = sub.add_parser("demo", help="load samples from a cache and print shapes")
    p.add_argument("cache")
    p.add_argument("--quat-order", default="wxyz", choices=["wxyz", "xyzw"])

    args = parser.parse_args()
    if args.cmd == "build-cache":
        build_cache(args.src, args.out, tuple(args.size), args.crop, args.camera)
    elif args.cmd == "check-quat":
        check_quat_order(args.src)
    elif args.cmd == "info":
        print_info(args.cache)
    elif args.cmd == "demo":
        demo(args.cache, args.quat_order)


if __name__ == "__main__":
    main()
