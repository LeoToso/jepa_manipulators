#!/usr/bin/env python
"""OGBench cube data -> the cache format used by the world models and the Diffusion Policy.

    # 1. get the data: download the official dataset ...
    python -m jepa.ogb_data download --dataset visual-cube-single-play-v0 --dir ~/.ogbench/data
    # ... or collect it yourself with OGBench's scripted 'play' oracle (same procedure as theirs)
    MUJOCO_GL=egl python -m jepa.ogb_data collect --env visual-cube-single-v0 --episodes 1000 \\
        --out ~/.ogbench/data/visual-cube-single-play-v0.npz
    # 2. convert
    MUJOCO_GL=egl python -m jepa.ogb_data convert --dataset visual-cube-single-play-v0 --dir ~/.ogbench/data \\
        --out /mnt/t7shield/cache_ogb_cube_single

The cache has the same layout as jepa.data.build_cache: images.npy (N, 64, 64, 3) uint8,
lowdim.npz, meta.json. Low-dimensional fields are computed from the recorded qpos / qvel with the
environment's own MuJoCo model, exactly as the environment reports them at test time:
    q, dq      arm joint positions / velocities (6)       = info['proprio/joint_pos' / 'joint_vel']
    gripper    gripper opening in [0, 1]                    = info['proprio/gripper_opening']
    x_ee       effector position (3) + yaw as quaternion    (labels; GCB state reward)
    obj_pos    cube positions (3 per cube)                  (labels; GCB state reward)
    actions    the recorded 5-d actions [dx, dy, dz, dyaw, dgripper] in [-1, 1] (action_mode: stored)
Splits: OGBench's train file -> train; its validation file -> val and test (half each).
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile

import numpy as np

# Headless rendering on a GPU server: MuJoCo must use EGL (not GLFW, which needs a display). Set it
# before MuJoCo is imported anywhere; an explicit MUJOCO_GL in the environment still wins.
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', os.environ['MUJOCO_GL'])


def _env(env_name, **kw):
    import gymnasium
    import ogbench.manipspace  # noqa: F401  (registers the environments)
    return gymnasium.make(env_name, **kw)


def env_name_of(dataset):
    """'visual-cube-single-play-v0' -> 'visual-cube-single-v0'."""
    parts = dataset.split('-')
    return '-'.join(parts[:-2] + parts[-1:])


def proprio_from_info(info):
    """The proprio vector of a live environment step, in the cache's [q, dq, gripper] order."""
    return np.concatenate([info['proprio/joint_pos'], info['proprio/joint_vel'],
                           info['proprio/gripper_opening']]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Download / collect
# --------------------------------------------------------------------------- #

def download(dataset, out_dir):
    import ogbench
    ogbench.download_datasets([dataset], os.path.expanduser(out_dir))


def collect(env_name, n_episodes, out, max_steps=1001, noise=0.1, noise_smoothing=0.5, seed=0):
    """OGBench's 'play' data generation for the cube environments (data_gen_scripts/
    generate_manipspace.py with --dataset_type=play), writing the same .npz files."""
    from collections import defaultdict

    from ogbench.manipspace.oracles.plan.cube_plan import CubePlanOracle
    np.random.seed(seed)
    env = _env(env_name, terminate_at_goal=False, mode='data_collection', max_episode_steps=max_steps)
    agent = CubePlanOracle(env=env, noise=noise, noise_smoothing=noise_smoothing)
    data = defaultdict(list)
    n_val = max(1, n_episodes // 10)
    train_steps = 0
    for ep in range(n_episodes + n_val):
        ob, info = env.reset(seed=seed + ep)
        p_stack = {'single': 0.0, 'double': np.random.uniform(0, 0.25), 'triple': np.random.uniform(0.05, 0.35),
                   'quadruple': np.random.uniform(0.1, 0.5)}.get(env_name.split('-')[-2], 0.0)
        agent.reset(ob, info)
        done, steps = False, 0
        while not done:
            action = np.clip(np.array(agent.select_action(ob, info)), -1, 1)
            nob, _, term, trunc, info = env.step(action)
            done = term or trunc
            if agent.done:
                aob, ainfo = env.unwrapped.set_new_target(p_stack=p_stack)
                agent.reset(aob, ainfo)
            data['observations'].append(ob)
            data['actions'].append(action)
            data['terminals'].append(done)
            data['qpos'].append(info['prev_qpos'])
            data['qvel'].append(info['prev_qvel'])
            ob = nob
            steps += 1
        if ep < n_episodes:
            train_steps += steps
        print(f'  episode {ep + 1}/{n_episodes + n_val}: {steps} steps', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    for path, sl in ((out, slice(0, train_steps)), (out.replace('.npz', '-val.npz'), slice(train_steps, None))):
        d = {k: np.array(v[sl], dtype=np.uint8 if (k == 'observations' and v[0].dtype == np.uint8)
                         else (bool if k == 'terminals' else np.float32)) for k, v in data.items()}
        np.savez_compressed(path, **d)
        print(f'wrote {path}: {len(d["actions"])} steps')


# --------------------------------------------------------------------------- #
# Convert
# --------------------------------------------------------------------------- #

def _read_header(fh):
    """(shape, fortran_order, dtype) of a .npy stream (public numpy API, any version)."""
    version = np.lib.format.read_magic(fh)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(fh)
    return np.lib.format.read_array_header_2_0(fh)


def _stream_observations(npz_path, out_mm, offset):
    """Copy 'observations' of an .npz into a memmap without loading it all into memory."""
    with zipfile.ZipFile(npz_path) as zf, zf.open('observations.npy') as fh:
        shape, fortran, dtype = _read_header(fh)
        assert not fortran and dtype == np.uint8, (shape, dtype)
        per = int(np.prod(shape[1:]))
        n, chunk = shape[0], 4096
        for i in range(0, n, chunk):
            m = min(chunk, n - i)
            buf = fh.read(m * per)
            out_mm[offset + i:offset + i + m] = np.frombuffer(buf, np.uint8).reshape(m, *shape[1:])
    return n


def _npz_shape(npz_path, key):
    with zipfile.ZipFile(npz_path) as zf, zf.open(f'{key}.npy') as fh:
        return _read_header(fh)[0]


def convert(dataset, data_dir, out, env_name=None, max_train_episodes=None, seed=0):
    data_dir = os.path.expanduser(data_dir)
    env_name = env_name or env_name_of(dataset)
    files = [os.path.join(data_dir, f'{dataset}.npz'), os.path.join(data_dir, f'{dataset}-val.npz')]
    for f in files:
        if not os.path.exists(f):
            raise SystemExit(f'missing {f}: run the download or collect command first')
    os.makedirs(out, exist_ok=True)

    # low-dimensional arrays (small), episode boundaries
    lows, lengths, which = [], [], []
    for fi, f in enumerate(files):
        z = np.load(f)
        if 'qpos' not in z.files:
            raise SystemExit(f'{f} has no qpos: robot state is needed for proprio')
        term = z['terminals'].astype(bool)
        ends = np.flatnonzero(term)
        if len(ends) == 0 or ends[-1] != len(term) - 1:
            ends = np.append(ends, len(term) - 1)
        ep_len = np.diff(np.concatenate([[-1], ends]))
        keep = len(ep_len) if (fi > 0 or max_train_episodes is None) else min(max_train_episodes, len(ep_len))
        n_keep = int(ep_len[:keep].sum())
        lows.append({k: z[k][:n_keep] for k in ('actions', 'qpos', 'qvel')})
        lengths.append(ep_len[:keep])
        which.append(np.full(keep, fi))
    lengths_all = np.concatenate(lengths)
    total = int(lengths_all.sum())
    print(f'{dataset}: {len(lengths_all)} episodes, {total} frames', flush=True)

    # images, streamed
    shape0 = _npz_shape(files[0], 'observations')
    images = np.lib.format.open_memmap(os.path.join(out, 'images.npy'), mode='w+', dtype=np.uint8,
                                       shape=(total, *shape0[1:]))
    off = 0
    for f, low in zip(files, lows):
        n_keep = len(low['actions'])
        tmp_n = _npz_shape(f, 'observations')[0]
        if tmp_n == n_keep:
            _stream_observations(f, images, off)
        else:  # truncated train set: stream into a scratch view and keep the first n_keep
            with zipfile.ZipFile(f) as zf, zf.open('observations.npy') as fh:
                shape, _, _ = _read_header(fh)
                per = int(np.prod(shape[1:]))
                for i in range(0, n_keep, 4096):
                    m = min(4096, n_keep - i)
                    images[off + i:off + i + m] = np.frombuffer(fh.read(m * per), np.uint8).reshape(m, *shape[1:])
        off += n_keep
        print(f'  images from {os.path.basename(f)}: {n_keep}', flush=True)
    images.flush()

    # proprio and labels with the environment's model (forward kinematics)
    import mujoco
    env = _env(env_name).unwrapped
    model, dat = env._model, env._data
    arm, grip_id, site = np.asarray(env._arm_joint_ids), env._gripper_opening_joint_id, env._pinch_site_id
    n_cubes = int(getattr(env, '_num_cubes', 1))
    obj_adr = [model.jnt_qposadr[model.joint(f'object_joint_{i}').id] for i in range(n_cubes)]
    qpos = np.concatenate([lw['qpos'] for lw in lows]).astype(np.float64)
    qvel = np.concatenate([lw['qvel'] for lw in lows]).astype(np.float64)
    ee = np.zeros((total, 3), np.float32)
    yaw = np.zeros(total, np.float32)
    for t in range(total):
        dat.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, dat)
        ee[t] = dat.site_xpos[site]
        m = dat.site_xmat[site].reshape(3, 3)
        yaw[t] = np.arctan2(m[1, 0], m[0, 0])
        if t % 200000 == 0:
            print(f'  kinematics {t}/{total}', flush=True)
    quat = np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], 1)
    actions = np.concatenate([lw['actions'] for lw in lows]).astype(np.float32)
    x_ee = np.concatenate([ee, quat], 1).astype(np.float32)
    gripper = np.clip(qpos[:, grip_id:grip_id + 1] / 0.8, 0, 1).astype(np.float32)
    starts = np.concatenate([[0], np.cumsum(lengths_all)[:-1]])
    np.savez(os.path.join(out, 'lowdim.npz'),
             episode_lengths=lengths_all.astype(np.int64), episode_starts=starts.astype(np.int64),
             episode_task=np.zeros(len(lengths_all), np.int64),
             actions=actions, q=qpos[:, arm].astype(np.float32), dq=qvel[:, arm].astype(np.float32),
             gripper=gripper, gripper_cmd=actions[:, 4:5], x_ee=x_ee, x_ee_des=x_ee,
             v_ee=np.zeros((total, 6), np.float32),
             obj_pos=np.concatenate([qpos[:, a:a + 3] for a in obj_adr], 1).astype(np.float32),
             timestamps=np.arange(total, dtype=np.float64) * float(env._control_timestep if hasattr(env, '_control_timestep') else 0.05))

    rng = np.random.RandomState(seed)
    allw = np.concatenate(which)
    train = np.flatnonzero(allw == 0).tolist()
    held = rng.permutation(np.flatnonzero(allw == 1)).tolist()
    meta = {'source': f'ogbench:{dataset}', 'env_name': env_name, 'image_size': int(shape0[1]),
            'tasks': ['cube_play'], 'num_frames': total, 'n_cubes': n_cubes,
            'splits': {'train': train, 'val': sorted(held[:(len(held) + 1) // 2]),
                       'test': sorted(held[(len(held) + 1) // 2:] or held[:1])}}
    with open(os.path.join(out, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    print('split (episodes): ' + ', '.join(f'{k}={len(v)}' for k, v in meta['splits'].items()))
    print(f'[done] {out}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('download')
    d.add_argument('--dataset', required=True)
    d.add_argument('--dir', default='~/.ogbench/data')
    c = sub.add_parser('collect')
    c.add_argument('--env', required=True, help='e.g. visual-cube-single-v0')
    c.add_argument('--episodes', type=int, default=1000)
    c.add_argument('--max-steps', type=int, default=1001)
    c.add_argument('--out', required=True, help='.npz path; the validation file is written next to it')
    c.add_argument('--seed', type=int, default=0)
    v = sub.add_parser('convert')
    v.add_argument('--dataset', required=True, help='e.g. visual-cube-single-play-v0')
    v.add_argument('--dir', default='~/.ogbench/data')
    v.add_argument('--out', required=True)
    v.add_argument('--max-train-episodes', type=int, default=None)
    a = p.parse_args()
    if a.cmd == 'download':
        download(a.dataset, a.dir)
    elif a.cmd == 'collect':
        collect(a.env, a.episodes, os.path.expanduser(a.out), a.max_steps, seed=a.seed)
    else:
        convert(a.dataset, a.dir, a.out, max_train_episodes=a.max_train_episodes)


if __name__ == '__main__':
    main()
