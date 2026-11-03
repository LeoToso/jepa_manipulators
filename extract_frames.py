#!/usr/bin/env python3
"""Save N evenly spaced, full-resolution camera frames from one episode.

The first saved frame is the episode's first frame, the last is its last
frame, and the rest are spread evenly in between. Frames come straight from
the raw .h5 file (672x376, no resizing or cropping) and are saved as
lossless PNGs, plus a contact sheet and a CSV with the matching robot state.

    python extract_frames.py Dataset/20261026_150630.h5 --episode 0
    python extract_frames.py Dataset/20261026_150630.h5 --episode 3 --num 10 --scale 2

Output (default frames/<file>_<episode>/):
    frame_00_step0000.png ... frame_09_step0435.png
    contact_sheet.png        all frames in a grid, labelled with step and time
    frames.csv               step, time, EE position, gripper for each frame

Requirements: h5py, numpy, pillow.
"""

import argparse
import csv
import os

import h5py
import numpy as np
from PIL import Image, ImageDraw


def pick_steps(length, num):
    """`num` step indices from 0 to length - 1 inclusive, evenly spaced."""
    if num > length:
        raise SystemExit(f"episode has only {length} frames, cannot pick {num}")
    return np.linspace(0, length - 1, num).round().astype(int)


def contact_sheet(images, labels, cols):
    w, h = images[0].size
    rows = int(np.ceil(len(images) / cols))
    pad, label_h = 6, 22
    sheet = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + label_h + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (img, label) in enumerate(zip(images, labels)):
        x = pad + (k % cols) * (w + pad)
        y = pad + (k // cols) * (h + label_h + pad)
        draw.text((x + 2, y + 4), label, fill="black")
        sheet.paste(img, (x, y + label_h))
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", help="an .h5 file from the Dataset folder")
    parser.add_argument("--episode", default="0", help="episode index (0, 1, ...) or full name")
    parser.add_argument("--num", type=int, default=10, help="number of frames (default 10)")
    parser.add_argument("--camera", default=None, help="camera serial (default: the first one)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="upscale saved frames by this factor (Lanczos), default 1 = original")
    parser.add_argument("--out-dir", default=None, help="default frames/<file>_<episode>/")
    parser.add_argument("--cols", type=int, default=5, help="contact sheet columns (default 5)")
    args = parser.parse_args()

    with h5py.File(args.file, "r") as f:
        names = sorted(k for k in f.keys() if isinstance(f[k], h5py.Group))
        if args.episode in names:
            name = args.episode
        else:
            try:
                name = names[int(args.episode)]
            except (ValueError, IndexError):
                raise SystemExit(f"episode '{args.episode}' not found; available: {', '.join(names)}")
        ep = f[name]
        cam = args.camera or sorted(ep["cams"].keys())[0]
        rgb = ep["cams"][cam]["rgb"]
        steps = pick_steps(rgb.shape[0], args.num)
        frames = [rgb[i] for i in steps]  # read only the frames we need
        t = ep["timestamps"][()]
        x_ee = ep["fr3/x_ee"][()]
        gripper = ep["fr3/gripper"][()]

    stem = os.path.splitext(os.path.basename(args.file))[0]
    out = args.out_dir or os.path.join("frames", f"{stem}_{name}")
    os.makedirs(out, exist_ok=True)

    images, labels = [], []
    for k, (step, frame) in enumerate(zip(steps, frames)):
        img = Image.fromarray(frame)
        if args.scale != 1.0:
            img = img.resize((round(img.width * args.scale), round(img.height * args.scale)),
                             Image.LANCZOS)
        img.save(os.path.join(out, f"frame_{k:02d}_step{step:04d}.png"))
        images.append(img)
        labels.append(f"#{k}  step {step}  t={t[step] - t[0]:.1f}s")

    contact_sheet(images, labels, args.cols).save(os.path.join(out, "contact_sheet.png"))

    with open(os.path.join(out, "frames.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "step", "time_s", "ee_x", "ee_y", "ee_z", "gripper"])
        for k, step in enumerate(steps):
            writer.writerow([k, int(step), f"{t[step] - t[0]:.3f}",
                             *(f"{v:.4f}" for v in x_ee[step, :3]), f"{gripper[step]:.3f}"])

    print(f"{stem} {name}: {rgb.shape[0]} frames, camera {cam}")
    print(f"saved {len(steps)} frames ({images[0].width}x{images[0].height}) at steps "
          f"{', '.join(str(s) for s in steps)}")
    print(f"-> {out}/")


if __name__ == "__main__":
    main()
