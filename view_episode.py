#!/usr/bin/env python3
"""Play back a Franka FR3 episode: camera images next to joint, end-effector
and gripper plots, so you can see what the robot did.

The server has no display, so the result is written to a file that you can
open in VS Code (or download):

    # video of one episode (.mp4 if ffmpeg is available, otherwise .gif)
    python view_episode.py Dataset/20261026_143452.h5 --episode 0

    # quick still overview (8 frames + plots) as a PNG
    python view_episode.py Dataset/20261026_143452.h5 --episode 0 --png

    # PNG overviews for every episode in a file
    python view_episode.py Dataset/20261026_143452.h5 --episode all --png

    # list the episodes in a file
    python view_episode.py Dataset/20261026_143452.h5 --list

For .mp4 output, matplotlib needs ffmpeg. If it is missing, either
`pip install imageio-ffmpeg` (no admin rights needed) or `conda install ffmpeg`.

Requirements: h5py, numpy, matplotlib.
"""

import argparse
import os

import h5py
import matplotlib

matplotlib.use("Agg")  # no display on the server
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

JOINT_COLORS = plt.cm.tab10(np.arange(7))
XYZ_COLORS = ["tab:red", "tab:green", "tab:blue"]


def list_episodes(f):
    return sorted(k for k in f.keys() if isinstance(f[k], h5py.Group))


def resolve_episodes(f, episode):
    names = list_episodes(f)
    if episode == "all":
        return names
    if episode in names:
        return [episode]
    try:
        return [names[int(episode)]]
    except (ValueError, IndexError):
        raise SystemExit(f"episode '{episode}' not found; available: {', '.join(names)}")


def camera_key(ep, camera=None):
    cams = sorted(ep["cams"].keys())
    if camera is None:
        return cams[0]
    if camera not in cams:
        raise SystemExit(f"camera '{camera}' not found; available: {', '.join(cams)}")
    return camera


def load_lowdim(ep):
    return {
        "t": ep["timestamps"][()],
        "q": ep["fr3/q"][()],
        "x_ee": ep["fr3/x_ee"][()],
        "x_ee_des": ep["action/x_ee_des"][()],
        "gripper": ep["fr3/gripper"][()],
        "gripper_cmd": ep["action/gripper"][()],
    }


def shared_x_axes(fig, specs):
    """Stacked subplots sharing the bottom one's time axis."""
    bottom = fig.add_subplot(specs[-1])
    axes = [fig.add_subplot(s, sharex=bottom) for s in specs[:-1]]
    for ax in axes:
        ax.tick_params(labelbottom=False)
    return axes + [bottom]


def draw_plots(axes, d):
    """Draw the time-series plots once; returns the moving cursor lines."""
    ax_q, ax_x, ax_g = axes
    t = d["t"]
    for j in range(7):
        ax_q.plot(t, d["q"][:, j], color=JOINT_COLORS[j], lw=1, label=f"J{j + 1}")
    ax_q.set_ylabel("joint angle [rad]")
    ax_q.legend(ncol=7, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.25), frameon=False)

    for i, name in enumerate("xyz"):
        ax_x.plot(t, d["x_ee"][:, i], color=XYZ_COLORS[i], lw=1.2, label=f"{name}")
        ax_x.plot(t, d["x_ee_des"][:, i], color=XYZ_COLORS[i], lw=1, ls="--")
    ax_x.set_ylabel("EE position [m]")
    ax_x.legend(ncol=3, fontsize=6, loc="best", title="solid=actual, dashed=target",
                title_fontsize=6)

    ax_g.plot(t, d["gripper"], color="black", lw=1.2, label="state")
    ax_g.plot(t, d["gripper_cmd"], color="tab:orange", lw=1, ls="--", label="command")
    ax_g.set_ylabel("gripper")
    ax_g.set_ylim(-1.15, 1.15)
    ax_g.set_xlabel("time [s]")
    ax_g.legend(fontsize=6, loc="best")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.yaxis.label.set_size(8)
    return [ax.axvline(t[0], color="gray", lw=1) for ax in axes]


def make_video(path, ep_name, out, every, speed, camera):
    with h5py.File(path, "r") as f:
        ep = f[ep_name]
        d = load_lowdim(ep)
        cam = camera_key(ep, camera)
        print(f"loading {ep['cams'][cam]['rgb'].shape[0]} frames from camera {cam} ...")
        frames = ep["cams"][cam]["rgb"][::every]
    idx = np.arange(0, len(d["t"]), every)

    fig = plt.figure(figsize=(12, 4.8), dpi=100)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.55, 1], hspace=0.45, wspace=0.18,
                          left=0.02, right=0.98, top=0.9, bottom=0.1)
    ax_img = fig.add_subplot(gs[:, 0])
    axes = shared_x_axes(fig, [gs[i, 1] for i in range(3)])
    im = ax_img.imshow(frames[0])
    ax_img.axis("off")
    title = fig.suptitle("", fontsize=10)
    cursors = draw_plots(axes, d)

    def update(k):
        i = idx[k]
        im.set_data(frames[k])
        for c in cursors:
            c.set_xdata([d["t"][i], d["t"][i]])
        title.set_text(f"{os.path.basename(path)}  {ep_name}  step {i}/{len(d['t']) - 1}  "
                       f"t={d['t'][i]:.1f}s")
        return [im, title] + cursors

    dt = np.median(np.diff(d["t"])) if len(d["t"]) > 1 else 0.1
    fps = max(1.0, speed / (dt * every))

    ext = os.path.splitext(out)[1].lower()
    writer = None
    if ext == ".mp4":
        writer = _ffmpeg_writer(fps)
        if writer is None:
            out = os.path.splitext(out)[0] + ".gif"
            print("ffmpeg not found -> writing a .gif instead "
                  "(pip install imageio-ffmpeg to get .mp4)")
    if writer is None:
        writer = animation.PillowWriter(fps=fps)

    anim = animation.FuncAnimation(fig, update, frames=len(idx), blit=False)
    print(f"rendering {len(idx)} frames at {fps:.1f} fps -> {out}")
    anim.save(out, writer=writer)
    plt.close(fig)
    print(f"saved {out}")


def _ffmpeg_writer(fps):
    if not animation.writers.is_available("ffmpeg"):
        try:
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return None
        if not animation.writers.is_available("ffmpeg"):
            return None
    return animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=3000,
                                  extra_args=["-pix_fmt", "yuv420p"])


def make_png(path, ep_name, out, n_thumbs, camera):
    with h5py.File(path, "r") as f:
        ep = f[ep_name]
        d = load_lowdim(ep)
        cam = camera_key(ep, camera)
        T = len(d["t"])
        picks = np.linspace(0, T - 1, n_thumbs).round().astype(int)
        thumbs = [ep["cams"][cam]["rgb"][i] for i in picks]

    fig = plt.figure(figsize=(2.0 * n_thumbs, 8.5), dpi=100)
    gs = fig.add_gridspec(4, n_thumbs, height_ratios=[1.1, 1, 1, 1], hspace=0.35, wspace=0.04,
                          left=0.05, right=0.99, top=0.93, bottom=0.06)
    for k, (i, img) in enumerate(zip(picks, thumbs)):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img)
        ax.set_title(f"step {i}  t={d['t'][i]:.1f}s", fontsize=7)
        ax.axis("off")
    axes = shared_x_axes(fig, [gs[r, :] for r in (1, 2, 3)])
    for c in draw_plots(axes, d):
        c.remove()
    for ax in axes:
        for i in picks:
            ax.axvline(d["t"][i], color="gray", lw=0.6, ls=":")
    fig.suptitle(f"{os.path.basename(path)}  {ep_name}  ({T} steps, {d['t'][-1]:.1f}s, "
                 f"camera {cam})", fontsize=10)
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", help="an .h5 file from the Dataset folder")
    parser.add_argument("--episode", default="0",
                        help="episode index (0, 1, ...), full name, or 'all' (default 0)")
    parser.add_argument("--list", action="store_true", help="list episodes and exit")
    parser.add_argument("--png", action="store_true",
                        help="write a still overview PNG instead of a video")
    parser.add_argument("--out-dir", default="episode_views", help="where to save (default episode_views/)")
    parser.add_argument("--every", type=int, default=1,
                        help="video: use every N-th frame (smaller, faster file)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="video: playback speed, 1.0 = real time")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4", help="video format")
    parser.add_argument("--thumbs", type=int, default=8, help="png: number of frames to show")
    parser.add_argument("--camera", default=None, help="camera serial (default: the first one)")
    args = parser.parse_args()

    with h5py.File(args.file, "r") as f:
        if args.list:
            for i, name in enumerate(list_episodes(f)):
                ep = f[name]
                t = ep["timestamps"][()]
                print(f"{i:3d}  {name}  {len(t)} steps  {t[-1] - t[0]:.1f}s  "
                      f"cams={sorted(ep['cams'].keys())}")
            return
        episodes = resolve_episodes(f, args.episode)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.file))[0]
    for ep_name in episodes:
        base = os.path.join(args.out_dir, f"{stem}_{ep_name}")
        if args.png:
            make_png(args.file, ep_name, base + ".png", args.thumbs, args.camera)
        else:
            make_video(args.file, ep_name, f"{base}.{args.format}", args.every, args.speed,
                       args.camera)


if __name__ == "__main__":
    main()
