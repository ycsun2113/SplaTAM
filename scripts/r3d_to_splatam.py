"""
Convert Record3D .r3d files to SplaTAM's nerfcapture dataset format.

The .r3d file is a ZIP archive containing:
  - rgbd/0.jpg, rgbd/1.jpg, ...   (RGB images)
  - rgbd/0.depth, rgbd/1.depth, ... (lzfse-compressed float32 depth maps, in meters)
  - rgbd/0.conf, rgbd/1.conf, ...   (confidence maps, optional)
  - metadata                        (JSON with intrinsics + per-frame poses)

Output format (SplaTAM nerfcapture):
  output_dir/
  ├── transforms.json
  ├── rgb/
  │   ├── 0.png, 1.png, ...
  └── depth/
      ├── 0.png, 1.png, ...

Usage:
  pip install pyliblzfse   # Required for depth decompression
  python scripts/r3d_to_splatam.py --input /path/to/recording.r3d --output ./experiments/iPhone_Captures/my_scan [--stride 1] [--max_frames -1]
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

try:
    import liblzfse
except ImportError:
    print("ERROR: pyliblzfse is required. Install it with: pip install pyliblzfse")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Record3D .r3d to SplaTAM nerfcapture format")
    parser.add_argument("--input", "-i", required=True, type=str, help="Path to .r3d file")
    parser.add_argument("--output", "-o", required=True, type=str, help="Output directory for the dataset")
    parser.add_argument("--depth_scale", type=float, default=10.0,
                        help="Depth scale factor (same as in SplaTAM config, default: 10.0)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Use every Nth frame (default: 1, use all frames)")
    parser.add_argument("--max_frames", type=int, default=-1,
                        help="Maximum number of frames to extract (-1 = all)")
    parser.add_argument("--min_confidence", type=int, default=0, choices=[0, 1, 2],
                        help="Minimum depth confidence level (0=low, 1=medium, 2=high). "
                             "Pixels below this confidence are set to 0 depth. Default: 0 (keep all)")
    return parser.parse_args()


def load_depth(depth_bytes, height, width):
    """Decompress and reshape an lzfse-compressed float32 depth map."""
    decompressed = liblzfse.decompress(depth_bytes)
    depth = np.frombuffer(decompressed, dtype=np.float32)
    # Record3D depth maps can be 256x192 (LiDAR) or other sizes
    # Try to infer the shape
    if depth.size == height * width:
        return depth.reshape((height, width))
    else:
        # Try common LiDAR resolutions
        for h, w in [(256, 192), (192, 256), (480, 640), (640, 480)]:
            if depth.size == h * w:
                return depth.reshape((h, w))
        raise ValueError(f"Cannot reshape depth of size {depth.size} to any known resolution")


def load_confidence(conf_bytes, height, width):
    """Decompress and reshape an lzfse-compressed uint8 confidence map."""
    decompressed = liblzfse.decompress(conf_bytes)
    conf = np.frombuffer(decompressed, dtype=np.uint8)
    if conf.size == height * width:
        return conf.reshape((height, width))
    else:
        for h, w in [(256, 192), (192, 256), (480, 640), (640, 480)]:
            if conf.size == h * w:
                return conf.reshape((h, w))
        raise ValueError(f"Cannot reshape confidence of size {conf.size} to any known resolution")


def quat_translation_to_matrix(pose_7):
    """Convert a 7-element [qx, qy, qz, qw, tx, ty, tz] pose to a 4x4 matrix."""
    qx, qy, qz, qw, tx, ty, tz = pose_7
    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def parse_poses(raw_poses):
    """Parse poses from Record3D metadata. Handles both 7-element (quat+trans)
    and 16-element (4x4 matrix) formats."""
    poses = []
    for pose_data in raw_poses:
        arr = np.array(pose_data, dtype=np.float64).flatten()
        if arr.size == 7:
            # [qx, qy, qz, qw, tx, ty, tz]
            poses.append(quat_translation_to_matrix(arr))
        elif arr.size == 16:
            poses.append(arr.reshape(4, 4))
        else:
            raise ValueError(f"Unexpected pose size: {arr.size} (expected 7 or 16)")
    return poses


def main():
    args = parse_args()

    r3d_path = Path(args.input)
    output_path = Path(args.output)

    if not r3d_path.exists():
        print(f"ERROR: Input file {r3d_path} does not exist")
        sys.exit(1)

    if output_path.exists():
        print(f"WARNING: Output directory {output_path} already exists. Files may be overwritten.")

    print(f"Reading {r3d_path}...")

    with zipfile.ZipFile(str(r3d_path), 'r') as zf:
        # Read metadata
        with zf.open('metadata') as f:
            metadata = json.load(f)

        # Extract intrinsics from K matrix
        # Record3D stores K in COLUMN-MAJOR order, so we need to transpose
        K_raw = np.array(metadata['K']).reshape(3, 3)
        # Detect column-major: if K[2,2]==1 it's row-major, if K[0,2]==0 and K[2,0]!=0 it's column-major
        if abs(K_raw[2, 0]) > abs(K_raw[0, 2]) or (K_raw[0, 2] == 0 and K_raw[2, 0] != 0):
            K = K_raw.T  # Transpose from column-major to row-major
            print("Detected column-major K matrix, transposing")
        else:
            K = K_raw
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        print(f"Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

        # Get list of available frames
        frame_files = sorted([
            name for name in zf.namelist()
            if name.startswith('rgbd/') and name.endswith('.jpg')
        ])
        frame_indices = sorted([
            int(Path(name).stem) for name in frame_files
        ])

        print(f"Found {len(frame_indices)} frames in the recording")

        # Get camera poses - Record3D stores them as either:
        #   - 7-element arrays: [qx, qy, qz, qw, tx, ty, tz]
        #   - 16-element arrays: flattened 4x4 matrix
        poses = None
        for key in ['poses', 'cameraPoses']:
            if key in metadata:
                raw_poses = metadata[key]
                sample_size = np.array(raw_poses[0]).flatten().size
                print(f"Found poses under '{key}' key (format: {sample_size}-element per pose)")
                poses = parse_poses(raw_poses)
                break

        if poses is not None:
            print(f"Found {len(poses)} camera poses")
        else:
            print("WARNING: No camera poses found in metadata. Using identity poses.")
            print("  SplaTAM can still run — it will estimate poses via tracking.")

        # Apply stride
        selected_indices = frame_indices[::args.stride]
        if args.max_frames > 0:
            selected_indices = selected_indices[:args.max_frames]

        print(f"Extracting {len(selected_indices)} frames (stride={args.stride})...")

        # Create output directories
        rgb_dir = output_path / "rgb"
        depth_dir = output_path / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        # Determine depth dimensions from first frame
        first_depth_name = f"rgbd/{frame_indices[0]}.depth"
        with zf.open(first_depth_name) as f:
            first_depth_bytes = f.read()
        decompressed = liblzfse.decompress(first_depth_bytes)
        depth_size = len(decompressed) // 4  # float32
        # Determine depth resolution
        depth_h, depth_w = None, None
        # Record3D LiDAR depth is 192 rows × 256 columns (height=192, width=256)
        for h, w in [(192, 256), (256, 192), (640, 480), (480, 640), (320, 240), (240, 320)]:
            if h * w == depth_size:
                depth_h, depth_w = h, w
                break
        if depth_h is None:
            print(f"ERROR: Unknown depth resolution (pixel count: {depth_size})")
            sys.exit(1)
        print(f"Depth resolution: {depth_w}x{depth_h} (h={depth_h}, w={depth_w})")

        # Read first RGB to get image dimensions
        with zf.open(f"rgbd/{frame_indices[0]}.jpg") as f:
            img_bytes = f.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        img_h, img_w = img.shape[:2]
        print(f"RGB resolution: {img_w}x{img_h}")

        # Scale intrinsics if depth and RGB resolutions differ
        # The intrinsics in metadata typically correspond to the RGB resolution
        # We save RGB at its original resolution and resize depth to match

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
            # --- RGB ---
            jpg_name = f"rgbd/{frame_idx}.jpg"
            with zf.open(jpg_name) as f:
                img_bytes = f.read()
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(str(rgb_dir / f"{out_idx}.png"), img)

            # --- Depth ---
            depth_name = f"rgbd/{frame_idx}.depth"
            with zf.open(depth_name) as f:
                depth_bytes = f.read()
            depth = load_depth(depth_bytes, depth_h, depth_w)  # float32, meters

            # Apply confidence mask if requested
            if args.min_confidence > 0:
                conf_name = f"rgbd/{frame_idx}.conf"
                try:
                    with zf.open(conf_name) as f:
                        conf_bytes = f.read()
                    conf = load_confidence(conf_bytes, depth_h, depth_w)
                    depth[conf < args.min_confidence] = 0.0
                except KeyError:
                    pass  # No confidence map available

            # Resize depth to match RGB resolution BEFORE converting to uint16
            # Use bilinear interpolation on float32 to avoid staircase banding artifacts
            # (LiDAR depth is 256x192, RGB is 1920x1440 — NN upscale causes severe banding)
            if depth.shape[0] != img_h or depth.shape[1] != img_w:
                depth = cv2.resize(depth, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

            # Convert depth from meters to uint16 with depth_scale encoding
            # (same encoding as nerfcapture2dataset.py: depth_uint16 = depth_meters * 65535 / depth_scale)
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth_uint16 = np.clip(depth * 65535.0 / float(args.depth_scale), 0, 65535).astype(np.uint16)
            cv2.imwrite(str(depth_dir / f"{out_idx}.png"), depth_uint16)

            # --- Pose ---
            if poses is not None and frame_idx < len(poses):
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
