#!/usr/bin/env python3
"""Inspect a robot (e.g. Franka) dataset and print what is inside it.

Works on a single file or a whole directory. The format is detected from the
file extension, and each format's library is imported only when needed, so
you only have to install what your dataset uses.

Supported formats:
    HDF5      .h5 .hdf5          (pip install h5py)       robomimic, many custom loggers
    Zarr      .zarr directories  (pip install zarr)       diffusion_policy-style datasets
    NumPy     .npy .npz          (numpy)
    Pickle    .pkl .pickle       (stdlib)
    PyTorch   .pt .pth           (pip install torch)
    JSON      .json .jsonl       (stdlib)
    Parquet   .parquet           (pip install pyarrow)    LeRobot datasets
    TFRecord  .tfrecord*         (pip install tensorflow) RLDS / Open-X / DROID
    Video     .mp4 .avi .mkv     (pip install opencv-python)
    ROS bags  .bag .mcap / ROS2 bag dirs (pip install rosbags)
    CSV       .csv               (pandas)

Usage:
    python inspect_dataset.py /path/to/dataset
    python inspect_dataset.py /path/to/episode_0.hdf5 --samples 3
    python inspect_dataset.py /path/to/dataset --files-per-type 5
"""

import argparse
import json
import os
import pickle
import signal
import sys
from collections import Counter, defaultdict

import numpy as np

INDENT = "  "

# Common Franka Panda dimensionalities, used only as hints.
FRANKA_DIM_HINTS = {
    7: "7 DoF -> joint positions/velocities/torques, or EE pose (xyz + quaternion)",
    8: "7 joints + gripper, or EE pose (xyz + quat) + gripper",
    9: "7 joints + 2 gripper fingers",
    6: "EE pose (xyz + euler) or EE twist / wrench",
    16: "4x4 homogeneous transform (e.g. libfranka O_T_EE, flattened)",
}


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #

def human_size(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def array_summary(arr, samples=0, max_stat_elems=5_000_000):
    """One-line description of an array plus optional stats and samples."""
    arr = np.asarray(arr)
    parts = [f"shape={arr.shape}", f"dtype={arr.dtype}"]
    lines = []

    if arr.size and np.issubdtype(arr.dtype, np.number) and arr.size <= max_stat_elems:
        a = arr.astype(np.float64)
        finite = np.isfinite(a)
        if finite.any():
            parts.append(f"min={a[finite].min():.4g} max={a[finite].max():.4g} "
                         f"mean={a[finite].mean():.4g}")
        if not finite.all():
            parts.append(f"non-finite={int((~finite).sum())}")
        # Per-dimension ranges for (T, D) signals like joint states/actions.
        if arr.ndim == 2 and arr.shape[1] <= 32:
            mins = np.nanmin(a, axis=0)
            maxs = np.nanmax(a, axis=0)
            lines.append("per-dim min: " + np.array2string(mins, precision=3, max_line_width=200))
            lines.append("per-dim max: " + np.array2string(maxs, precision=3, max_line_width=200))
    elif arr.size > max_stat_elems:
        parts.append("(stats skipped: too large)")

    if arr.ndim >= 1 and arr.shape[-1] in FRANKA_DIM_HINTS and arr.ndim <= 2:
        lines.append(f"hint: last dim {arr.shape[-1]} = {FRANKA_DIM_HINTS[arr.shape[-1]]}")
    if arr.ndim in (3, 4) and arr.dtype == np.uint8 and arr.shape[-1] in (1, 3, 4):
        lines.append("hint: looks like image data (..., H, W, C)")

    if samples and arr.ndim >= 1 and arr.size:
        with np.printoptions(precision=4, suppress=True, linewidth=200, threshold=50):
            for i in range(min(samples, arr.shape[0])):
                if arr.ndim <= 2:
                    lines.append(f"[{i}] {arr[i]}")
                else:
                    lines.append(f"[{i}] <{arr[i].shape} {arr.dtype}>")
    return ", ".join(parts), lines


def describe(obj, depth=0, samples=0, max_depth=8, max_items=30, key="root"):
    """Recursively print the structure of nested python objects."""
    pad = INDENT * depth
    if depth > max_depth:
        print(f"{pad}{key}: ... (max depth reached)")
        return

    if isinstance(obj, np.ndarray) or _is_tensor(obj):
        arr = _to_numpy(obj)
        head, lines = array_summary(arr, samples)
        print(f"{pad}{key}: array {head}")
        for line in lines:
            print(f"{pad}{INDENT}{line}")
    elif isinstance(obj, dict):
        print(f"{pad}{key}: dict ({len(obj)} keys)")
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                print(f"{pad}{INDENT}... {len(obj) - max_items} more keys")
                break
            describe(v, depth + 1, samples, max_depth, max_items, key=repr(k))
    elif isinstance(obj, (list, tuple)):
        kind = type(obj).__name__
        print(f"{pad}{key}: {kind} (len={len(obj)})")
        if not obj:
            return
        # Lists of numbers are better shown as arrays.
        if all(isinstance(x, (int, float, bool, np.number)) for x in obj[:100]):
            head, lines = array_summary(np.array(obj), samples)
            print(f"{pad}{INDENT}as array: {head}")
            for line in lines:
                print(f"{pad}{INDENT * 2}{line}")
            return
        describe(obj[0], depth + 1, samples, max_depth, max_items, key="[0]")
        types = Counter(type(x).__name__ for x in obj)
        if len(types) > 1:
            print(f"{pad}{INDENT}element types: {dict(types)}")
    elif isinstance(obj, (str, bytes)):
        text = obj if len(obj) <= 80 else obj[:80] + (b"..." if isinstance(obj, bytes) else "...")
        print(f"{pad}{key}: {type(obj).__name__} (len={len(obj)}) {text!r}")
    else:
        print(f"{pad}{key}: {type(obj).__name__} = {obj!r}"[:200])


def _is_tensor(obj):
    return type(obj).__module__.startswith(("torch", "tensorflow"))


def _to_numpy(obj):
    if hasattr(obj, "detach"):
        return obj.detach().cpu().numpy()
    if hasattr(obj, "numpy"):
        return obj.numpy()
    return np.asarray(obj)


# --------------------------------------------------------------------------- #
# Format-specific inspectors
# --------------------------------------------------------------------------- #

def inspect_hdf5(path, samples):
    import h5py

    def show_attrs(obj, pad):
        for k, v in obj.attrs.items():
            text = repr(v if not isinstance(v, bytes) else v.decode(errors="replace"))
            print(f"{pad}@{k} = {text[:200]}")

    with h5py.File(path, "r") as f:
        show_attrs(f, "")
        seen_groups = Counter()

        def visit(name, obj, depth=0):
            pad = INDENT * depth
            base = name.split("/")[-1] or "/"
            if isinstance(obj, h5py.Group):
                # Collapse long runs of sibling episodes like demo_0, demo_1, ...
                children = list(obj.keys())
                print(f"{pad}{base}/ (group, {len(children)} children)")
                show_attrs(obj, pad + INDENT)
                prefixes = Counter(c.rstrip("0123456789") for c in children)
                for c in children:
                    prefix = c.rstrip("0123456789")
                    if prefixes[prefix] > 3 and prefix != c:
                        seen_groups[(name, prefix)] += 1
                        if seen_groups[(name, prefix)] == 2:
                            print(f"{pad}{INDENT}... {prefixes[prefix] - 1} more '{prefix}*' "
                                  f"entries with (probably) the same layout")
                        if seen_groups[(name, prefix)] > 1:
                            continue
                    visit(f"{name}/{c}", obj[c], depth + 1)
            else:
                if obj.size * obj.dtype.itemsize <= 200_000_000:
                    head, lines = array_summary(obj[()], samples)
                else:
                    head, lines = f"shape={obj.shape}, dtype={obj.dtype} (too large to load)", []
                print(f"{pad}{base}: {head}")
                show_attrs(obj, pad + INDENT)
                for line in lines:
                    print(f"{pad}{INDENT}{line}")

        for k in f.keys():
            visit(k, f[k])


def inspect_zarr(path, samples):
    import zarr

    root = zarr.open(path, mode="r")

    def visit(node, name, depth):
        pad = INDENT * depth
        if hasattr(node, "shape") and hasattr(node, "dtype") and not hasattr(node, "keys"):
            nbytes = int(np.prod(node.shape)) * node.dtype.itemsize
            if nbytes <= 200_000_000:
                head, lines = array_summary(node[...], samples)
            else:
                head, lines = f"shape={node.shape}, dtype={node.dtype} (too large to load)", []
            print(f"{pad}{name}: {head}")
            for line in lines:
                print(f"{pad}{INDENT}{line}")
        else:
            keys = sorted(node.keys())
            print(f"{pad}{name}/ (group, {len(keys)} children)")
            attrs = dict(node.attrs)
            if attrs:
                print(f"{pad}{INDENT}@attrs = {str(attrs)[:200]}")
            for k in keys:
                visit(node[k], k, depth + 1)

    visit(root, os.path.basename(path.rstrip("/")), 0)


def inspect_numpy(path, samples):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        describe({k: data[k] for k in data.files}, samples=samples)
    elif data.dtype == object and data.shape == ():
        describe(data.item(), samples=samples)
    else:
        describe(data, samples=samples)


def inspect_pickle(path, samples):
    with open(path, "rb") as f:
        describe(pickle.load(f), samples=samples)


def inspect_torch(path, samples):
    import torch
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only argument
        obj = torch.load(path, map_location="cpu")
    describe(obj, samples=samples)


def inspect_json(path, samples):
    if path.endswith(".jsonl"):
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        print(f"{len(rows)} lines")
        describe(rows, samples=samples)
    else:
        with open(path) as f:
            describe(json.load(f), samples=samples)


def inspect_parquet(path, samples):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    print(f"rows={pf.metadata.num_rows}, row_groups={pf.num_row_groups}")
    print("schema:")
    for field in pf.schema_arrow:
        print(f"{INDENT}{field.name}: {field.type}")
    table = pf.read()
    for name in table.column_names:
        col = table.column(name).to_pylist()
        if col and isinstance(col[0], (list, int, float)):
            try:
                head, lines = array_summary(np.array(col, dtype=float), samples)
                print(f"{INDENT}{name}: {head}")
                for line in lines:
                    print(f"{INDENT * 2}{line}")
            except (ValueError, TypeError):
                pass


def inspect_csv(path, samples):
    import pandas as pd
    df = pd.read_csv(path)
    print(f"rows={len(df)}, columns={len(df.columns)}")
    print(df.describe(include="all").T.to_string())
    if samples:
        print(df.head(samples).to_string())


def inspect_tfrecord(path, samples):
    import tensorflow as tf

    ds = tf.data.TFRecordDataset(path)
    count = 0
    for raw in ds:
        if count == 0:
            ex = tf.train.Example()
            ex.ParseFromString(raw.numpy())
            print("features of first record:")
            for name, feat in sorted(ex.features.feature.items()):
                kind = feat.WhichOneof("kind")
                values = getattr(feat, kind).value
                if kind == "bytes_list":
                    print(f"{INDENT}{name}: bytes x{len(values)} (first {len(values[0]) if values else 0} B)")
                else:
                    head, lines = array_summary(np.array(values), samples)
                    print(f"{INDENT}{name}: {kind} {head}")
                    for line in lines:
                        print(f"{INDENT * 2}{line}")
        count += 1
    print(f"records in file: {count}")


def inspect_video(path, samples):
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("could not open video (codec may be unsupported, e.g. AV1 in LeRobot)")
        return
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = frames / fps if fps else float("nan")
    print(f"{w}x{h}, {frames} frames, {fps:.2f} fps, {dur:.1f} s")
    cap.release()


def inspect_rosbag(path, samples):
    from rosbags.highlevel import AnyReader
    from pathlib import Path

    with AnyReader([Path(path)]) as reader:
        dur = (reader.end_time - reader.start_time) / 1e9
        print(f"duration={dur:.1f} s, messages={reader.message_count}")
        for conn in sorted(reader.connections, key=lambda c: c.topic):
            hz = conn.msgcount / dur if dur > 0 else 0
            print(f"{INDENT}{conn.topic}  [{conn.msgtype}]  {conn.msgcount} msgs (~{hz:.1f} Hz)")
        if samples:
            shown = set()
            for conn, _, raw in reader.messages():
                if conn.topic in shown:
                    continue
                shown.add(conn.topic)
                msg = reader.deserialize(raw, conn.msgtype)
                print(f"--- first message on {conn.topic}")
                print(INDENT + str(msg)[:600])


INSPECTORS = {
    ".h5": inspect_hdf5, ".hdf5": inspect_hdf5, ".hdf": inspect_hdf5,
    ".zarr": inspect_zarr,
    ".npy": inspect_numpy, ".npz": inspect_numpy,
    ".pkl": inspect_pickle, ".pickle": inspect_pickle,
    ".pt": inspect_torch, ".pth": inspect_torch,
    ".json": inspect_json, ".jsonl": inspect_json,
    ".parquet": inspect_parquet,
    ".csv": inspect_csv,
    ".tfrecord": inspect_tfrecord,
    ".mp4": inspect_video, ".avi": inspect_video, ".mkv": inspect_video,
    ".bag": inspect_rosbag, ".mcap": inspect_rosbag,
}


def file_kind(path):
    name = path.lower().rstrip("/")
    if ".tfrecord" in name:
        return ".tfrecord"
    if os.path.isdir(path):
        if name.endswith(".zarr") or os.path.exists(os.path.join(path, ".zgroup")):
            return ".zarr"
        if os.path.exists(os.path.join(path, "metadata.yaml")):
            return ".mcap"  # ROS 2 bag directory
        return None
    return os.path.splitext(name)[1]


def inspect_file(path, samples):
    kind = file_kind(path)
    size = os.path.getsize(path) if os.path.isfile(path) else dir_size(path)
    print(f"\n=== {path}  ({human_size(size)})")
    fn = INSPECTORS.get(kind)
    if fn is None:
        print(f"no inspector for '{kind}' files")
        return
    try:
        fn(path, samples)
    except ImportError as e:
        print(f"missing library for {kind}: {e}  (see install hints at top of script)")
    except Exception as e:  # keep going through the rest of the dataset
        print(f"failed to read: {type(e).__name__}: {e}")


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


# --------------------------------------------------------------------------- #
# Directory overview
# --------------------------------------------------------------------------- #

def inspect_directory(path, samples, files_per_type):
    by_kind = defaultdict(list)
    sizes = Counter()
    for root, dirs, files in os.walk(path):
        # Treat zarr stores and ROS 2 bag dirs as single "files".
        for d in list(dirs):
            full = os.path.join(root, d)
            kind = file_kind(full)
            if kind in (".zarr", ".mcap"):
                by_kind[kind].append(full)
                sizes[kind] += dir_size(full)
                dirs.remove(d)
        for f in files:
            full = os.path.join(root, f)
            kind = file_kind(full) or "(no ext)"
            by_kind[kind].append(full)
            try:
                sizes[kind] += os.path.getsize(full)
            except OSError:
                pass

    print(f"Dataset directory: {path}")
    print(f"total size: {human_size(sum(sizes.values()))}")
    print("file types:")
    for kind, paths in sorted(by_kind.items(), key=lambda kv: -sizes[kv[0]]):
        print(f"{INDENT}{kind:12s} {len(paths):7d} files  {human_size(sizes[kind]):>10s}")

    # Recognise well-known layouts.
    if os.path.exists(os.path.join(path, "meta", "info.json")):
        print("\nlooks like a LeRobot dataset; meta/info.json:")
        inspect_json(os.path.join(path, "meta", "info.json"), 0)
    if os.path.exists(os.path.join(path, "dataset_info.json")) or os.path.exists(
            os.path.join(path, "features.json")):
        print("\nlooks like an RLDS / TFDS dataset (dataset_info.json / features.json)")

    for kind, paths in sorted(by_kind.items()):
        if kind not in INSPECTORS:
            continue
        paths = sorted(paths)
        for p in paths[:files_per_type]:
            inspect_file(p, samples)
        if len(paths) > files_per_type:
            print(f"\n... skipped {len(paths) - files_per_type} more {kind} files "
                  f"(use --files-per-type to see more)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="dataset file or directory")
    parser.add_argument("--samples", type=int, default=0,
                        help="print the first N entries of each array (default 0)")
    parser.add_argument("--files-per-type", type=int, default=2,
                        help="in a directory, how many files of each type to open (default 2)")
    args = parser.parse_args()
    if hasattr(signal, "SIGPIPE"):  # exit quietly when piped into `head`
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    if not os.path.exists(args.path):
        sys.exit(f"path not found: {args.path}")
    if os.path.isdir(args.path) and file_kind(args.path) is None:
        inspect_directory(args.path, args.samples, args.files_per_type)
    else:
        inspect_file(args.path, args.samples)


if __name__ == "__main__":
    main()
