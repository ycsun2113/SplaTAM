"""
Convert Stray Scanner dataset to SplaTAM's nerfcapture dataset format.

Stray Scanner export structure:
  dataset_hash/
  ├── camera_matrix.csv    (3x3 intrinsic matrix)
  ├── odometry.csv         (timestamp, frame, x, y, z, qx, qy, qz, qw)
  ├── rgb.mp4              (HEVC video)
  ├── depth/               (16-bit PNG, 192x256, values in mm)
  │   ├── 000000.png
  │   └── ...
  └── confidence/          (8-bit PNG, 192x256, values 0/1/2)
      ├── 000000.png
      └── ...

Output format (SplaTAM nerfcapture):
  output_dir/
  ├── transforms.json
  ├── rgb/
  │   ├── 0.png, 1.png, ...
  └── depth/
      ├── 0.png, 1.png, ...

Usage:
  python scripts/stray_to_splatam.py --input /path/to/stray_dataset/ --output ./experiments/iPhone_Captures/my_scan [--stride 1] [--max_frames -1]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_exclude_frames(spec):
    """Parse a comma-separated list of frame ranges into a set of frame indices.
    Examples: '127-135,143-158' -> {127,128,...,135,143,...,158}
              '5,10,20-25'      -> {5,10,20,21,22,23,24,25}
    """
    excluded = set()
    if not spec:
        return excluded
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-', 1)
            excluded.update(range(int(lo), int(hi) + 1))
        else:
            excluded.add(int(part))
    return excluded


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Stray Scanner dataset to SplaTAM nerfcapture format")
    parser.add_argument("--input", "-i", required=True, type=str, help="Path to Stray Scanner dataset folder")
    parser.add_argument("--output", "-o", required=True, type=str, help="Output directory for the dataset")
    parser.add_argument("--depth_scale", type=float, default=10.0,
                        help="Depth scale factor (same as in SplaTAM config, default: 10.0)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Use every Nth frame (default: 1, use all frames)")
    parser.add_argument("--max_frames", type=int, default=-1,
                        help="Maximum number of frames to extract (-1 = all)")
    parser.add_argument("--min_confidence", type=int, default=0, choices=[0, 1, 2],
                        help="Minimum depth confidence level (0=low, 1=medium, 2=high). Default: 0 (keep all)")
    parser.add_argument("--exclude_frames", type=str, default="",
                        help="Comma-separated ranges of original frame indices to exclude, "
                             "e.g. '127-135,143-158'")
    return parser.parse_args()


def quat_translation_to_matrix(qx, qy, qz, qw, tx, ty, tz):
    """Convert quaternion + translation to a 4x4 transformation matrix."""
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def load_intrinsics(camera_matrix_path):
    """Load 3x3 intrinsic matrix from camera_matrix.csv."""
    K = np.loadtxt(camera_matrix_path, delimiter=',')
    assert K.shape == (3, 3), f"Expected 3x3 intrinsic matrix, got {K.shape}"
    return K


def load_odometry(odometry_path):
    """Load per-frame camera poses from odometry.csv.
    Returns dict: frame_idx -> 4x4 matrix.
    """
    poses = {}
    with open(odometry_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header if present
        # Check if first row is a header
        if header and not header[0].replace('.', '').replace('-', '').isdigit():
            pass  # It was a header, already consumed
        else:
            # First row is data, process it
            if header:
                row = header
                ts, frame = float(row[0]), int(row[1])
                x, y, z = float(row[2]), float(row[3]), float(row[4])
                qx, qy, qz, qw = float(row[5]), float(row[6]), float(row[7]), float(row[8])
                poses[frame] = quat_translation_to_matrix(qx, qy, qz, qw, x, y, z)

        for row in reader:
            if len(row) < 9:
                continue
            ts, frame = float(row[0]), int(row[1])
            x, y, z = float(row[2]), float(row[3]), float(row[4])
            qx, qy, qz, qw = float(row[5]), float(row[6]), float(row[7]), float(row[8])
            poses[frame] = quat_translation_to_matrix(qx, qy, qz, qw, x, y, z)
    return poses


def extract_rgb_frames(video_path, frame_indices, rgb_dir):
    """Extract specific frames from rgb.mp4 and save as PNGs.
    Returns (img_h, img_w) of the extracted frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        sys.exit(1)

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_video_frames} frames")

    # Build a set of frame indices we need
    needed = set(frame_indices)
    frame_map = {idx: out_idx for out_idx, idx in enumerate(frame_indices)}

    img_h, img_w = None, None
    current_frame = 0
    extracted = 0

    while cap.isOpened() and extracted < len(needed):
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame in needed:
            if img_h is None:
                img_h, img_w = frame.shape[:2]
            cv2.imwrite(str(rgb_dir / f"{frame_map[current_frame]}.png"), frame)
            extracted += 1
        current_frame += 1

    cap.release()
    print(f"Extracted {extracted} RGB frames")
    return img_h, img_w


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Validate input
    required_files = ['camera_matrix.csv', 'odometry.csv', 'rgb.mp4']
    required_dirs = ['depth']
    for f in required_files:
        if not (input_path / f).exists():
            print(f"ERROR: Missing required file: {input_path / f}")
            sys.exit(1)
    for d in required_dirs:
        if not (input_path / d).is_dir():
            print(f"ERROR: Missing required directory: {input_path / d}")
            sys.exit(1)

    if output_path.exists():
        print(f"WARNING: Output directory {output_path} already exists. Files may be overwritten.")

    print(f"Reading Stray Scanner dataset from {input_path}...")

    # Load intrinsics
    K = load_intrinsics(input_path / 'camera_matrix.csv')
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print(f"Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # Load poses
    poses = load_odometry(input_path / 'odometry.csv')
    print(f"Loaded {len(poses)} camera poses")

    # Get available depth frames
    depth_files = sorted((input_path / 'depth').glob('*.png'))
    frame_indices = sorted([int(f.stem) for f in depth_files])
    print(f"Found {len(frame_indices)} depth frames")

    # Apply stride and max_frames
    selected_indices = frame_indices[::args.stride]
    if args.max_frames > 0:
        selected_indices = selected_indices[:args.max_frames]

    # Exclude problematic frames by output index (position after stride)
    exclude_set = parse_exclude_frames(args.exclude_frames)
    if exclude_set:
        before = len(selected_indices)
        selected_indices = [frame for idx, frame in enumerate(selected_indices) if idx not in exclude_set]
        print(f"Excluded {before - len(selected_indices)} frames (output indices: {args.exclude_frames})")

    print(f"Selecting {len(selected_indices)} frames (stride={args.stride})")

    # Create output directories
    rgb_dir = output_path / "rgb"
    depth_out_dir = output_path / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_out_dir.mkdir(parents=True, exist_ok=True)

    # Extract RGB frames from video
    print("Extracting RGB frames from video...")
    img_h, img_w = extract_rgb_frames(input_path / 'rgb.mp4', selected_indices, rgb_dir)
    if img_h is None:
        print("ERROR: No RGB frames were extracted")
        sys.exit(1)
    print(f"RGB resolution: {img_w}x{img_h}")

    # Get depth resolution from first frame
    sample_depth = cv2.imread(str(input_path / 'depth' / f'{frame_indices[0]:06d}.png'), cv2.IMREAD_UNCHANGED)
    depth_h, depth_w = sample_depth.shape[:2]
    print(f"Depth resolution: {depth_w}x{depth_h}")

    # Check if confidence maps are available
    has_confidence = (input_path / 'confidence').is_dir()

    # Build manifest
    manifest = {
        "fl_x": float(fx),
        "fl_y": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "w": int(img_w),
        "h": int(img_h),
        "integer_depth_scale": float(args.depth_scale) / 65535.0,
        "frames": []
    }

    for out_idx, frame_idx in enumerate(selected_indices):
        # --- Depth ---
        depth_path = input_path / 'depth' / f'{frame_idx:06d}.png'
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)  # 16-bit, values in mm

        # Apply confidence mask if requested
        if args.min_confidence > 0 and has_confidence:
            conf_path = input_path / 'confidence' / f'{frame_idx:06d}.png'
            if conf_path.exists():
                conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
                depth[conf < args.min_confidence] = 0

        # Convert from mm to meters, then to SplaTAM's uint16 encoding
        # Stray depth is in mm (uint16), convert: depth_m = depth_mm / 1000.0
        # SplaTAM encoding: depth_uint16 = depth_m * 65535 / depth_scale
        depth_m = depth.astype(np.float32) / 1000.0
        depth_uint16 = (depth_m * 65535.0 / float(args.depth_scale)).astype(np.uint16)

        # Resize depth to match RGB resolution
        depth_uint16 = cv2.resize(depth_uint16, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(depth_out_dir / f"{out_idx}.png"), depth_uint16)

        # --- Pose ---
        if frame_idx in poses:
            c2w = poses[frame_idx].astype(np.float32)
        else:
            c2w = np.eye(4, dtype=np.float32)

        frame_entry = {
            "transform_matrix": c2w.tolist(),
            "file_path": f"rgb/{out_idx}.png",
            "fl_x": float(fx),
            "fl_y": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "w": int(img_w),
            "h": int(img_h),
            "depth_path": f"depth/{out_idx}.png"
        }
        manifest["frames"].append(frame_entry)

        if (out_idx + 1) % 10 == 0 or out_idx == len(selected_indices) - 1:
            print(f"  Processed {out_idx + 1}/{len(selected_indices)} frames")

    # Save manifest
    manifest_path = output_path / "transforms.json"
    with open(str(manifest_path), "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"\nDone! Dataset saved to: {output_path}")
    print(f"  - {len(selected_indices)} frames extracted")
    print(f"  - RGB: {img_w}x{img_h}")
    print(f"  - Depth scale: {args.depth_scale}")
    print(f"\nTo run SplaTAM, update your config:")
    print(f'  base_dir = "{output_path.parent}"')
    print(f'  scene_name = "{output_path.name}"')
    print(f'  num_frames = {len(selected_indices)}')
    print(f'  full_res_width = {img_w}')
    print(f'  full_res_height = {img_h}')


if __name__ == "__main__":
    main()
