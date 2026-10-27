# jepa_manipulators

Tools for the Franka FR3 demonstration data in `Dataset/*.h5`.

## Data layout

Each `.h5` file holds episodes `episode_0000000, episode_0000001, ...`, recorded at about 10 Hz.
All streams within an episode share the same timestamps.

| Key | Shape | Meaning |
|---|---|---|
| `cams/<serial>/rgb` | (T, 376, 672, 3) uint8 | camera images (gzip-compressed) |
| `fr3/q`, `fr3/dq` | (T, 7) | joint angles [rad], joint velocities [rad/s] |
| `fr3/x_ee` | (T, 7) | end-effector position [m] + orientation quaternion |
| `fr3/v_ee` | (T, 6) | end-effector velocity: linear [m/s] + angular [rad/s] |
| `fr3/gripper` | (T,) | gripper state, -1 to 1 |
| `action/x_ee_des` | (T, 7) | target end-effector pose (the action) |
| `action/gripper` | (T,) | gripper command, -1 to 1 |
| `timestamps` | (T,) | seconds since the start of the episode |

## Scripts

```bash
# what is in a dataset (any format)
python inspect_dataset.py Dataset

# watch an episode: video, or a still PNG overview
python view_episode.py Dataset/20261026_143452.h5 --list
python view_episode.py Dataset/20261026_143452.h5 --episode 0
python view_episode.py Dataset/20261026_143452.h5 --episode all --png

# training data
python franka_dataset.py check-quat Dataset                      # find the quaternion order
python franka_dataset.py build-cache Dataset --out cache_224 --size 224 224
python franka_dataset.py demo cache_224 --quat-order wxyz
```

```python
from franka_dataset import FrankaDataset, split_episodes

train_eps, val_eps = split_episodes("cache_224", val_fraction=0.15)
train = FrankaDataset("cache_224", episodes=train_eps, action_mode="delta", quat_order="wxyz")
val = FrankaDataset("cache_224", episodes=val_eps, action_mode="delta", quat_order="wxyz",
                    stats=train.stats)
```

See the docstring at the top of `franka_dataset.py` for all options: action modes, multi-frame windows, normalization.

Requirements: `numpy h5py matplotlib pillow torch`. For `.mp4` videos, also install ffmpeg: `pip install imageio-ffmpeg`.
