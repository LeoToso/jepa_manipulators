"""Data pipeline for the multi-task Franka FR3 dataset.

1. ``build_cache`` reads the merged recording file (``dataset_franka_arm.h5``;
   episodes ``episode_0000000 ...`` with root attributes ``tasks`` and
   ``episodes_per_task``), resizes every camera frame once and writes

       <cache>/images.npy   (N, S, S, 3) uint8, memory-mapped at train time
       <cache>/lowdim.npz   robot state, per-step actions, episode/task index
       <cache>/meta.json    task names, image size, train/val/test episode split

   The split is made once, per task, so every model sees the same episodes.

2. ``WindowDataset`` samples windows of ``horizon + 1`` latent steps, where one
   latent step is ``frame_skip`` recorded steps (10 Hz -> 10/frame_skip Hz).
   The action of a latent step is the chunk of the ``frame_skip`` per-step
   actions it spans, each [dpos (3), drot (3), gripper command (1)].

   Per-step action modes:
       delta_next    achieved motion: pose[t+1] - pose[t]   (default)
       delta_target  commanded offset: x_ee_des[t] - pose[t]
   Rotations are rotation vectors of q_to * q_from^-1 in the robot base frame;
   quaternions are stored w, x, y, z (verified on this robot's logs).

   Optionally a hindsight goal is returned: the frame k latent steps after the
   window start, k ~ Geometric(goal_geom_p) truncated to the episode, with the
   measured state of the next frame and of the goal. It is used by the
   goal-conditioned bisimulation loss.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

LOWDIM_KEYS = {  # cache key -> key inside an episode group
    'x_ee': 'fr3/x_ee', 'x_ee_des': 'action/x_ee_des', 'q': 'fr3/q', 'dq': 'fr3/dq',
    'v_ee': 'fr3/v_ee', 'gripper': 'fr3/gripper', 'gripper_cmd': 'action/gripper',
    'timestamps': 'timestamps',
}
ACTION_DIM = 7  # dpos (3) + drot (3) + gripper command (1), per recorded step


# --------------------------------------------------------------------------- #
# Quaternions (numpy, w-x-y-z)
# --------------------------------------------------------------------------- #

def to_wxyz(q, order):
    if order == 'wxyz':
        return q
    if order == 'xyzw':
        return np.roll(q, 1, axis=-1)
    raise ValueError(f'quat_order must be wxyz or xyzw, got {order!r}')


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
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, q[..., :1])
    return v * np.where(s > 1e-8, angle / np.maximum(s, 1e-8), 2.0)


def relative_rotvec(q_from, q_to):
    """Base-frame rotation vector taking orientation q_from to q_to."""
    return quat_to_rotvec(quat_mul(q_to, quat_conj(q_from)))


def per_step_actions(x_ee, x_ee_des, gripper_cmd, mode, quat_order='wxyz'):
    """(T, 7) per-step actions of one episode; the last row has no successor
    in delta_next mode and is never used as an action."""
    q = to_wxyz(x_ee[:, 3:7].astype(np.float64), quat_order)
    if mode == 'delta_next':
        pos_to = np.concatenate([x_ee[1:, :3], x_ee[-1:, :3]])
        q_to = np.concatenate([q[1:], q[-1:]])
    elif mode == 'delta_target':
        pos_to = x_ee_des[:, :3]
        q_to = to_wxyz(x_ee_des[:, 3:7].astype(np.float64), quat_order)
    else:
        raise ValueError(f'unknown action_mode {mode!r}')
    return np.concatenate([pos_to - x_ee[:, :3], relative_rotvec(q, q_to),
                           gripper_cmd.reshape(-1, 1)], axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Cache building
# --------------------------------------------------------------------------- #

def _decode(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def parse_task_names(attr):
    """The `tasks` root attribute as a list of names. Accepts an array of strings, or a
    single string holding a JSON list ('["a", "b"]') or comma-separated names."""
    if attr is None:
        return []
    if isinstance(attr, (bytes, str, np.bytes_, np.str_)) or np.ndim(attr) == 0:
        text = _decode(attr.item() if hasattr(attr, 'item') else attr).strip()
        if text.startswith('['):
            try:
                return [str(t) for t in json.loads(text)]
            except ValueError:
                text = text.strip('[]')
        return [t.strip().strip('\'"') for t in text.split(',') if t.strip()]
    return [_decode(t) for t in attr]


def _episode_task(name, group, episodes_per_task, task_names):
    for key in ('task_id', 'task'):
        if key in group.attrs:
            value = group.attrs[key]
            if isinstance(value, (bytes, str)):
                value = _decode(value)
                if value in task_names:
                    return task_names.index(value)
                return int(value)
            return int(value)
    return int(name.split('_')[-1]) // int(episodes_per_task)


def _resize(frames, size):
    from PIL import Image
    out = np.empty((len(frames), size, size, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        out[i] = np.asarray(Image.fromarray(frame).resize((size, size), Image.BILINEAR))
    return out


def build_cache(h5_path, out, image_size=128, camera=None, val_per_task=5,
                test_per_task=5, seed=0, chunk=64, max_episodes=None):
    """Resize all frames once and store everything needed for training/eval."""
    import h5py
    os.makedirs(out, exist_ok=True)
    with h5py.File(h5_path, 'r') as f:
        task_names = parse_task_names(f.attrs.get('tasks'))
        ept = int(f.attrs.get('episodes_per_task', 0)) or None
        names = sorted(k for k in f.keys() if isinstance(f[k], h5py.Group))
        if max_episodes:
            names = names[:max_episodes]
        if ept is None and not task_names:
            raise ValueError('file has neither a `tasks` nor an `episodes_per_task` attribute')
        lengths, tasks, cams = [], [], []
        for name in names:
            ep = f[name]
            cam = camera or sorted(ep['cams'].keys())[0]
            lengths.append(int(ep['cams'][cam]['rgb'].shape[0]))
            tasks.append(_episode_task(name, ep, ept or 1, task_names))
            cams.append(cam)
        if not task_names:
            task_names = [f'task_{i}' for i in range(max(tasks) + 1)]
        if max(tasks) >= len(task_names):
            raise ValueError(f'task id {max(tasks)} but only {len(task_names)} task names: {task_names}')
        print('tasks:', task_names)
        total = int(sum(lengths))
        print(f'{len(names)} episodes, {total} frames, {len(task_names)} tasks -> {out} '
              f'(images {image_size}x{image_size}: {total * image_size ** 2 * 3 / 1e9:.1f} GB)')

        images = np.lib.format.open_memmap(os.path.join(out, 'images.npy'), mode='w+',
                                           dtype=np.uint8, shape=(total, image_size, image_size, 3))
        low = {k: [] for k in LOWDIM_KEYS}
        pos = 0
        for i, (name, n, cam) in enumerate(zip(names, lengths, cams)):
            ep = f[name]
            rgb = ep['cams'][cam]['rgb']
            for s in range(0, n, chunk):
                frames = rgb[s:s + chunk]
                images[pos + s:pos + s + len(frames)] = _resize(frames, image_size)
            for key, src in LOWDIM_KEYS.items():
                arr = ep[src][()]
                if len(arr) != n:
                    raise ValueError(f'{name}/{src} has {len(arr)} rows, camera has {n}')
                low[key].append(arr.reshape(n, -1) if key != 'timestamps' else arr)
            pos += n
            print(f'  [{i + 1}/{len(names)}] {name} task={task_names[tasks[i]]} {n} steps',
                  flush=True)
        images.flush()

    lengths = np.asarray(lengths, dtype=np.int64)
    tasks = np.asarray(tasks, dtype=np.int64)
    np.savez(os.path.join(out, 'lowdim.npz'),
             episode_lengths=lengths,
             episode_starts=np.concatenate([[0], np.cumsum(lengths)[:-1]]),
             episode_task=tasks,
             **{k: np.concatenate(v).astype(np.float64 if k == 'timestamps' else np.float32)
                for k, v in low.items()})

    rng = np.random.RandomState(seed)
    splits = {'train': [], 'val': [], 'test': []}
    for t in range(len(task_names)):
        eps = rng.permutation(np.flatnonzero(tasks == t)).tolist()
        n_val = min(val_per_task, max(len(eps) - 2, 0))
        n_test = min(test_per_task, max(len(eps) - n_val - 1, 0))
        splits['val'] += sorted(eps[:n_val])
        splits['test'] += sorted(eps[n_val:n_val + n_test])
        splits['train'] += sorted(eps[n_val + n_test:])
    meta = {
        'source': os.path.abspath(h5_path), 'image_size': int(image_size),
        'tasks': task_names, 'episode_names': names, 'cameras': cams,
        'num_frames': total, 'splits': {k: sorted(v) for k, v in splits.items()},
        'split_seed': int(seed),
    }
    with open(os.path.join(out, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    print('split (episodes): ' + ', '.join(f'{k}={len(v)}' for k, v in splits.items()))
    return meta


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def task_family(name):
    """'pick_and_place_banana' -> 'pick_and_place', 'rotate_chips' -> 'rotate'."""
    return name.rsplit('_', 1)[0] if '_' in name else name


def task_object(name):
    return name.rsplit('_', 1)[-1]


class Cache:
    """Everything except the images, held in memory (a few hundred MB at most)."""

    def __init__(self, cache_dir, quat_order='wxyz'):
        self.dir = cache_dir
        with open(os.path.join(cache_dir, 'meta.json')) as fh:
            self.meta = json.load(fh)
        low = np.load(os.path.join(cache_dir, 'lowdim.npz'))
        self.low = {k: low[k] for k in low.files}
        self.lengths = self.low['episode_lengths']
        self.starts = self.low['episode_starts']
        self.episode_task = self.low['episode_task']
        self.tasks = self.meta['tasks']
        self.families = sorted(set(task_family(t) for t in self.tasks))
        self.objects = sorted(set(task_object(t) for t in self.tasks))
        self.quat_order = quat_order
        self.episode_of_frame = np.repeat(np.arange(len(self.lengths)), self.lengths)
        self.step_in_episode = np.concatenate([np.arange(n) for n in self.lengths])
        self._images = None

    @property
    def images(self):
        if self._images is None:  # opened lazily so each DataLoader worker maps it itself
            self._images = np.load(os.path.join(self.dir, 'images.npy'), mmap_mode='r')
        return self._images

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_images'] = None
        return state

    def split_episodes(self, split, tasks=None):
        eps = list(self.meta['splits'][split])
        if tasks is not None:
            ids = [self.tasks.index(t) if isinstance(t, str) else int(t) for t in tasks]
            eps = [e for e in eps if self.episode_task[e] in ids]
        return eps

    def actions(self, mode):
        """(N, 7) per-step actions for every frame, computed episode by episode."""
        out = np.zeros((len(self.episode_of_frame), ACTION_DIM), dtype=np.float32)
        for e, (s, n) in enumerate(zip(self.starts, self.lengths)):
            sl = slice(s, s + n)
            out[sl] = per_step_actions(self.low['x_ee'][sl], self.low['x_ee_des'][sl],
                                       self.low['gripper_cmd'][sl, 0], mode, self.quat_order)
        return out


def action_stats(cache, mode, episodes):
    acts = cache.actions(mode)
    rows = np.concatenate([np.arange(cache.starts[e], cache.starts[e] + cache.lengths[e] - 1)
                           for e in episodes])
    a = acts[rows].astype(np.float64)
    return {'mean': a.mean(0).astype(np.float32), 'std': (a.std(0) + 1e-6).astype(np.float32)}


class WindowDataset(Dataset):
    """Windows of horizon+1 latent steps from the episodes of one split.

    Returns a dict:
        obs        uint8 (H+1, 3, S, S)  frames t, t+fs, ..., t+H*fs
        prev_obs   uint8 (H+1, 3, S, S)  the frame fs steps before each (for frame
                                         differencing; clamped at the episode start)
        actions    float (H, 7*fs)       normalized action chunks
        frame      int   (H+1,)          global frame index of every step (labels/probes)
        task, episode  int
        goal_obs, goal_prev_obs, goal_offset, next_state, goal_state   (if return_goal)
    """

    def __init__(self, cache, split, horizon=3, frame_skip=2, action_mode='delta_next',
                 stats=None, return_goal=False, goal_geom_p=0.3, goal_max_offset=30,
                 tasks=None, start_stride=1, seed=0):
        self.cache = cache
        self.H, self.fs = int(horizon), int(frame_skip)
        self.return_goal = bool(return_goal)
        self.goal_geom_p = float(goal_geom_p)
        self.goal_max_offset = int(goal_max_offset)
        self.episodes = cache.split_episodes(split, tasks)
        if not self.episodes:
            raise ValueError(f'no episodes in split {split!r} for tasks {tasks}')
        self.stats = stats or action_stats(cache, action_mode, cache.split_episodes('train', tasks))
        raw = cache.actions(action_mode)
        self.actions = ((raw - self.stats['mean']) / self.stats['std']).astype(np.float32)
        span = self.H * self.fs  # last frame index offset; needs an action up to span-1
        starts = []
        for e in self.episodes:
            s, n = cache.starts[e], cache.lengths[e]
            if n - 1 >= span:
                starts.append(np.arange(s, s + n - 1 - span + 1, int(start_stride)))
        if not starts:
            raise ValueError('episodes are shorter than horizon * frame_skip')
        self.index = np.concatenate(starts).astype(np.int64)
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.index)

    def _state(self, frame):
        """Measured [EE position (3), quaternion w-x-y-z (4), gripper (1)] of one frame."""
        low = self.cache.low
        q = to_wxyz(low['x_ee'][frame, 3:7].astype(np.float64), self.cache.quat_order)
        return torch.from_numpy(np.concatenate([low['x_ee'][frame, :3], q, low['gripper'][frame, :1]])
                                .astype(np.float32))

    def _frames(self, idx):
        return torch.from_numpy(np.ascontiguousarray(self.cache.images[idx])).permute(0, 3, 1, 2)

    def __getitem__(self, i):
        c = self.cache
        start = int(self.index[i])
        ep = int(c.episode_of_frame[start])
        ep_start = int(c.starts[ep])
        ep_last = ep_start + int(c.lengths[ep]) - 1
        frames = start + np.arange(self.H + 1) * self.fs
        prev = np.maximum(frames - self.fs, ep_start)
        act_rows = start + np.arange(self.H * self.fs)
        actions = self.actions[act_rows].reshape(self.H, self.fs * self.actions.shape[1])
        out = {
            'obs': self._frames(frames), 'prev_obs': self._frames(prev),
            'actions': torch.from_numpy(actions), 'frame': torch.from_numpy(frames),
            'task': int(c.episode_task[ep]), 'episode': ep,
        }
        if self.return_goal:
            k_avail = (ep_last - start) // self.fs
            k = min(int(self.rng.geometric(self.goal_geom_p)), self.goal_max_offset, k_avail)
            g = start + k * self.fs
            out['goal_obs'] = self._frames(np.array([g]))[0]
            out['goal_prev_obs'] = self._frames(np.array([g - self.fs]))[0]
            out['goal_offset'] = k
            out['goal_frame'] = g
            out['next_state'] = self._state(frames[1])
            out['goal_state'] = self._state(g)
        return out


def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.RandomState((torch.initial_seed() + worker_id) % 2 ** 32)


def update_task_names(h5_path, out):
    """Rewrite only the task names in <out>/meta.json from the dataset's `tasks` attribute
    (task ids and the episode split are unchanged; no images are touched)."""
    import h5py
    with h5py.File(h5_path, 'r') as f:
        names = parse_task_names(f.attrs.get('tasks'))
    path = os.path.join(out, 'meta.json')
    with open(path) as fh:
        meta = json.load(fh)
    n_tasks = int(np.load(os.path.join(out, 'lowdim.npz'))['episode_task'].max()) + 1
    if len(names) != n_tasks:
        raise ValueError(f'{len(names)} task names but the cache has {n_tasks} task ids: {names}')
    meta['tasks'] = names
    with open(path, 'w') as fh:
        json.dump(meta, fh, indent=1)
    print('tasks:', names)
    print('split (episodes): ' + ', '.join(f'{k}={len(v)}' for k, v in meta['splits'].items()))


def main():
    import argparse
    p = argparse.ArgumentParser(description='Build the training cache from dataset_franka_arm.h5')
    p.add_argument('h5', help='merged dataset file (e.g. dataset_franka_arm.h5)')
    p.add_argument('--out', required=True, help='cache folder')
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--camera', default=None)
    p.add_argument('--val-per-task', type=int, default=5)
    p.add_argument('--test-per-task', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-episodes', type=int, default=None, help='debug: only the first N episodes')
    p.add_argument('--meta-only', action='store_true',
                   help='only rewrite the task names in an existing cache (no images)')
    a = p.parse_args()
    if a.meta_only:
        return update_task_names(a.h5, a.out)
    build_cache(a.h5, a.out, a.image_size, a.camera, a.val_per_task, a.test_per_task, a.seed,
                max_episodes=a.max_episodes)


if __name__ == '__main__':
    main()
