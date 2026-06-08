#!/usr/bin/env python3
"""
Convert VGGT-SLAM TUM-format poses to nerfstudio and/or COLMAP format.

VGGT-SLAM outputs poses in TUM format:
    frame_id tx ty tz qx qy qz qw
where (tx, ty, tz) is the camera-to-world translation and
(qx, qy, qz, qw) is the camera-to-world rotation quaternion
in OpenCV camera convention (+Y down, +Z forward).

This script:
1. Reads the VGGT-SLAM pose file
2. Deduplicates frame IDs (keeps last occurrence from overlapping submaps)
3. Converts each pose to a 4x4 c2w matrix
4. Copies/symlinks corresponding images from the sub10 image folder
5. Generates downsampled image copies (images_2, images_4, images_8)
6. Writes output in nerfstudio (transforms.json) and/or COLMAP (sparse/0/) format

Nerfstudio output:
    output_dir/
    ├── transforms.json
    ├── images/
    ├── images_2/
    ├── images_4/
    └── images_8/

COLMAP output:
    output_dir/
    ├── images/
    ├── images_2/, images_4/, images_8/
    └── sparse/0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
"""

import argparse
import json
import os
import shutil
import struct
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser(description="Convert VGGT-SLAM poses to nerfstudio/COLMAP format")
    parser.add_argument("--pose_file", type=str, required=True,
                        help="Path to VGGT-SLAM TUM pose file")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Path to the image directory (containing 0.png, 1.png, ...)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for formatted data")
    parser.add_argument("--format", type=str, choices=["nerfstudio", "colmap", "both"],
                        default="both", help="Output format (default: both)")
    parser.add_argument("--fl_x", type=float, default=1346.5076904296875,
                        help="Focal length x")
    parser.add_argument("--fl_y", type=float, default=1346.5076904296875,
                        help="Focal length y")
    parser.add_argument("--cx", type=float, default=957.650634765625,
                        help="Principal point x")
    parser.add_argument("--cy", type=float, default=707.76336669921875,
                        help="Principal point y")
    parser.add_argument("--w", type=int, default=1920, help="Image width")
    parser.add_argument("--h", type=int, default=1440, help="Image height")
    parser.add_argument("--camera_model", type=str, default="OPENCV",
                        help="Camera model for nerfstudio")
    parser.add_argument("--symlink", action="store_true",
                        help="Use symlinks instead of copying images")
    parser.add_argument("--downsample_factors", type=int, nargs="*", default=[2, 4, 8],
                        help="Downsample factors for generating images_2, images_4, etc.")
    parser.add_argument("--npz_dir", type=str, default=None,
                        help="Path to VGGT-SLAM dense point cloud logs (*.npz files) "
                             "for populating points3D.bin")
    parser.add_argument("--max_points", type=int, default=100_000,
                        help="Max number of sparse points to keep (default: 100000)")
    parser.add_argument("--voxel_size", type=float, default=0.02,
                        help="Voxel size for downsampling points (default: 0.02m)")
    return parser.parse_args()


# --------------------------------------------------------------------------
# Pose I/O
# --------------------------------------------------------------------------

def read_vggt_poses(pose_file):
    """
    Read VGGT-SLAM TUM-format pose file.
    Returns OrderedDict: frame_id -> (tx, ty, tz, qx, qy, qz, qw)
    Keeps the LAST occurrence of each frame_id (later submaps have more context).
    """
    poses = OrderedDict()
    with open(pose_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            frame_id = int(float(parts[0]))
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            poses[frame_id] = (tx, ty, tz, qx, qy, qz, qw)
    return poses


def tum_to_c2w_matrix(tx, ty, tz, qx, qy, qz, qw):
    """Convert TUM pose to 4x4 c2w matrix. Quat is (qx,qy,qz,qw) scipy order."""
    rot = Rotation.from_quat([qx, qy, qz, qw])
    c2w = np.eye(4)
    c2w[:3, :3] = rot.as_matrix()
    c2w[:3, 3] = [tx, ty, tz]
    return c2w


# --------------------------------------------------------------------------
# COLMAP binary writers
# --------------------------------------------------------------------------

def write_cameras_bin(path, camera_id, model_id, width, height, params):
    """Write COLMAP cameras.bin. model_id: 1=PINHOLE (fx,fy,cx,cy)."""
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', 1))  # num cameras
        f.write(struct.pack('<I', camera_id))
        f.write(struct.pack('<i', model_id))
        f.write(struct.pack('<QQ', width, height))
        for p in params:
            f.write(struct.pack('<d', p))


def write_images_bin(path, images_data):
    """Write COLMAP images.bin. Each entry: w2c quaternion & translation."""
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(images_data)))
        for img in images_data:
            f.write(struct.pack('<I', img['id']))
            f.write(struct.pack('<dddd', img['qw'], img['qx'], img['qy'], img['qz']))
            f.write(struct.pack('<ddd', img['tx'], img['ty'], img['tz']))
            f.write(struct.pack('<I', img['camera_id']))
            name_bytes = img['name'].encode('utf-8') + b'\x00'
            f.write(name_bytes)
            f.write(struct.pack('<Q', 0))  # num 2D points = 0


def write_points3d_bin(path, points=None, colors=None):
    """
    Write COLMAP points3D.bin.
    points: (N,3) float64, colors: (N,3) uint8. If None, writes empty file.
    """
    with open(path, 'wb') as f:
        if points is None or len(points) == 0:
            f.write(struct.pack('<Q', 0))
            return
        f.write(struct.pack('<Q', len(points)))
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i] if colors is not None else (128, 128, 128)
            f.write(struct.pack('<Q', i + 1))    # point3D_id
            f.write(struct.pack('<ddd', x, y, z))
            f.write(struct.pack('<BBB', int(r), int(g), int(b)))
            f.write(struct.pack('<d', 0.0))       # error
            f.write(struct.pack('<Q', 0))          # track length = 0


def load_vggt_pointclouds(npz_dir, image_dir, max_points=100_000, voxel_size=0.02):
    """
    Load VGGT-SLAM dense point clouds from per-frame .npz files.
    Each npz has 'pointcloud' (H,W,3) and 'mask' (H,W).
    Points are already in VGGT-SLAM world frame.
    Returns (points, colors) after voxel downsampling.
    """
    npz_files = sorted(
        [f for f in os.listdir(npz_dir) if f.endswith('.npz')],
        key=lambda x: float(x.replace('.npz', ''))
    )
    print(f"  Found {len(npz_files)} npz files in {npz_dir}")

    # Sample a subset of frames for efficiency
    step = max(1, len(npz_files) // 50)  # use ~50 frames
    npz_subset = npz_files[::step]
    print(f"  Using {len(npz_subset)} frames (every {step}th)")

    all_points = []
    all_colors = []
    image_dir = Path(image_dir)

    for npz_name in npz_subset:
        data = np.load(os.path.join(npz_dir, npz_name))
        pointcloud = data['pointcloud']  # (H, W, 3)
        mask = data['mask']              # (H, W)
        H, W = pointcloud.shape[:2]

        pts = pointcloud[mask]  # (M, 3)
        # Filter invalid points
        valid = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) < 100.0)
        pts = pts[valid]

        # Try to get colors from corresponding image
        frame_id = npz_name.replace('.npz', '')
        # Remove trailing .0 if present
        if frame_id.endswith('.0'):
            frame_id = frame_id[:-2]
        img_path = None
        for ext in ['.png', '.jpg']:
            p = image_dir / f"{frame_id}{ext}"
            if p.exists():
                img_path = p
                break

        if img_path is not None:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            colors = img[mask][valid]
            all_colors.append(colors)
        else:
            all_colors.append(np.full((len(pts), 3), 128, dtype=np.uint8))

        all_points.append(pts)

    if not all_points:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    print(f"  Total raw points: {len(all_points):,}")

    # Voxel downsample using simple grid-based approach (no open3d dependency)
    if voxel_size > 0 and len(all_points) > 0:
        # Quantize to voxel grid
        voxel_indices = np.floor(all_points / voxel_size).astype(np.int64)
        # Use unique voxels - keep first point per voxel
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
        all_points = all_points[unique_idx]
        all_colors = all_colors[unique_idx]
        print(f"  After voxel downsample ({voxel_size}m): {len(all_points):,}")

    # Limit total points
    if len(all_points) > max_points:
        idx = np.random.RandomState(42).choice(len(all_points), max_points, replace=False)
        all_points = all_points[idx]
        all_colors = all_colors[idx]
        print(f"  After subsampling to max_points: {len(all_points):,}")

    return all_points.astype(np.float64), all_colors.astype(np.uint8)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()

    # Read poses
    print(f"Reading poses from {args.pose_file}...")
    poses = read_vggt_poses(args.pose_file)
    print(f"  Found {len(poses)} unique frames (after deduplication)")

    # Create output directories
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for factor in args.downsample_factors:
        (output_dir / f"images_{factor}").mkdir(parents=True, exist_ok=True)

    # Process each frame
    ns_frames = []           # nerfstudio frame entries
    colmap_images_data = []  # COLMAP image entries
    image_names_out = []
    skipped = 0
    image_dir = Path(args.image_dir)

    # OpenCV -> OpenGL axis flip (for nerfstudio c2w)
    opencv_to_opengl = np.diag([1.0, -1.0, -1.0, 1.0])

    for idx, (frame_id, pose_data) in enumerate(sorted(poses.items())):
        tx, ty, tz, qx, qy, qz, qw = pose_data

        # Find source image
        src_image = image_dir / f"{frame_id}.png"
        if not src_image.exists():
            src_image = image_dir / f"{frame_id}.jpg"
        if not src_image.exists():
            print(f"  WARNING: Image not found for frame {frame_id}, skipping")
            skipped += 1
            continue

        # Output image name
        ext = src_image.suffix
        out_name = f"frame_{idx + 1:05d}{ext}"
        image_names_out.append(out_name)

        # Copy or symlink full-res image
        dst_image = images_dir / out_name
        if not dst_image.exists():
            if args.symlink:
                dst_image.symlink_to(src_image.resolve())
            else:
                shutil.copy2(src_image, dst_image)

        # Generate downsampled copies
        img = cv2.imread(str(src_image))
        for factor in args.downsample_factors:
            ds_dir = output_dir / f"images_{factor}"
            ds_path = ds_dir / out_name
            if not ds_path.exists():
                new_w = args.w // factor
                new_h = args.h // factor
                ds_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(ds_path), ds_img)

        # Build c2w in OpenCV convention (as output by VGGT-SLAM)
        c2w_opencv = tum_to_c2w_matrix(tx, ty, tz, qx, qy, qz, qw)

        # --- Nerfstudio format: c2w in OpenGL convention ---
        c2w_opengl = c2w_opencv @ opencv_to_opengl
        ns_frames.append({
            "file_path": f"images/{out_name}",
            "transform_matrix": c2w_opengl.tolist(),
        })

        # --- COLMAP format: w2c in OpenCV convention ---
        # COLMAP stores w2c rotation as quaternion (qw,qx,qy,qz) and translation
        rot_c2w = Rotation.from_quat([qx, qy, qz, qw])
        rot_w2c = rot_c2w.inv()
        quat_w2c = rot_w2c.as_quat()  # [qx, qy, qz, qw]
        camera_center = np.array([tx, ty, tz])
        t_w2c = -rot_w2c.as_matrix() @ camera_center

        colmap_images_data.append({
            'id': idx + 1,
            'qw': quat_w2c[3],
            'qx': quat_w2c[0],
            'qy': quat_w2c[1],
            'qz': quat_w2c[2],
            'tx': t_w2c[0],
            'ty': t_w2c[1],
            'tz': t_w2c[2],
            'camera_id': 1,
            'name': out_name,
        })

    print(f"  Processed {len(ns_frames)} frames, skipped {skipped}")

    # --- Write nerfstudio format ---
    if args.format in ("nerfstudio", "both"):
        transforms = {
            "fl_x": args.fl_x,
            "fl_y": args.fl_y,
            "cx": args.cx,
            "cy": args.cy,
            "w": args.w,
            "h": args.h,
            "camera_model": args.camera_model,
            "frames": ns_frames,
        }
        out_path = output_dir / "transforms.json"
        with open(out_path, "w") as f:
            json.dump(transforms, f, indent=2)
        print(f"\n[nerfstudio] transforms.json: {len(ns_frames)} frames")

    # --- Write COLMAP format ---
    if args.format in ("colmap", "both"):
        sparse_dir = output_dir / "sparse" / "0"
        sparse_dir.mkdir(parents=True, exist_ok=True)

        # cameras.bin: PINHOLE model (id=1), params = [fx, fy, cx, cy]
        write_cameras_bin(
            str(sparse_dir / "cameras.bin"),
            camera_id=1, model_id=1,
            width=args.w, height=args.h,
            params=[args.fl_x, args.fl_y, args.cx, args.cy],
        )
        write_images_bin(str(sparse_dir / "images.bin"), colmap_images_data)

        # Load sparse 3D points from VGGT-SLAM dense point clouds
        points3d, colors3d = None, None
        if args.npz_dir and os.path.isdir(args.npz_dir):
            print(f"\nLoading point clouds from {args.npz_dir}...")
            points3d, colors3d = load_vggt_pointclouds(
                args.npz_dir, args.image_dir,
                max_points=args.max_points, voxel_size=args.voxel_size,
            )

        write_points3d_bin(str(sparse_dir / "points3D.bin"), points3d, colors3d)
        n_pts = len(points3d) if points3d is not None else 0
        print(f"\n[COLMAP] sparse/0/: cameras.bin, images.bin ({len(colmap_images_data)} images), points3D.bin ({n_pts:,} points)")

    # Summary
    print(f"\nOutput written to {output_dir}")
    print(f"  images/: {len(ns_frames)} images")
    for factor in args.downsample_factors:
        ds_count = len(list((output_dir / f"images_{factor}").iterdir()))
        print(f"  images_{factor}/: {ds_count} images")

    if args.format in ("nerfstudio", "both"):
        print(f"\n  ns-train splatfacto --data {output_dir}")
    if args.format in ("colmap", "both"):
        print(f"\n  python simple_trainer.py --data_dir {output_dir} --data_factor 4")


if __name__ == "__main__":
    main()
