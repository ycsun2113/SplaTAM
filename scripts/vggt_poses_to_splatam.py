"""
Convert VGGT-SLAM poses to SplaTAM's nerfcapture transforms.json format.

This allows running SplaTAM's gaussian_splatting.py with VGGT-SLAM estimated poses
instead of ARKit poses or SplaTAM's own tracking poses.

Usage:
    # 1. Convert VGGT-SLAM poses to transforms.json
    python scripts/vggt_poses_to_splatam.py \
        --poses_file /path/to/VGGT-SLAM/outputs/scan_poses.txt \
        --transforms_json /path/to/dataset/transforms.json \
        --output /path/to/dataset/transforms_vggt.json

    # 2. Then run SplaTAM's Gaussian Splatting with the new poses:
    #    - Copy transforms_vggt.json -> transforms.json (backup the original first)
    #    - Run: python scripts/gaussian_splatting.py configs/iphone/gaussian_splatting.py
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert VGGT-SLAM poses to SplaTAM nerfcapture transforms.json"
    )
    parser.add_argument("--poses_file", required=True, type=str,
                        help="Path to VGGT-SLAM poses .txt file")
    parser.add_argument("--transforms_json", required=True, type=str,
                        help="Path to existing SplaTAM transforms.json "
                             "(for intrinsics and frame file paths)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output transforms.json path. "
                             "Default: transforms_vggt.json in same dir as input")
    return parser.parse_args()


def read_vggt_poses(poses_file):
    """
    Read VGGT-SLAM poses file.
    Format per line: frame_id tx ty tz qx qy qz qw

    From VGGT-SLAM's decompose_camera():
        R = inv(R_rq), t = -R @ inv(K) @ P[:,3]
    This gives the world-to-camera rotation R and translation t
    such that X_cam = R @ X_world + t.
    """
    poses = {}
    # P flips Y and Z axes: converts between OpenCV and OpenGL conventions
    # SplaTAM's nerfcapture.py applies P @ c2w @ P.T when reading transforms.json,
    # expecting c2w in OpenGL convention (Y-up, Z-backward).
    # VGGT-SLAM's decompose_camera outputs in OpenCV convention (Y-down, Z-forward).
    # So we need to convert: c2w_opengl = P @ c2w_opencv @ P
    P = np.array([[1, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]], dtype=np.float64)

    with open(poses_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            values = [float(v) for v in line.split()]
            if len(values) != 8:
                continue
            frame_id = int(values[0])
            tx, ty, tz = values[1], values[2], values[3]
            qx, qy, qz, qw = values[4], values[5], values[6], values[7]

            # VGGT-SLAM decompose_camera(no_inverse=False) outputs:
            #   R = inv(R_rq) = R_c2w (camera-to-world rotation)
            #   t = camera center in world frame
            # So the quaternion encodes R_c2w and (tx,ty,tz) is the camera position.
            
            # Build camera-to-world (c2w) matrix directly in OpenCV convention
            rot_c2w = R.from_quat([qx, qy, qz, qw])  # scipy uses [x, y, z, w]
            c2w_opencv = np.eye(4, dtype=np.float64)
            c2w_opencv[:3, :3] = rot_c2w.as_matrix()
            c2w_opencv[:3, 3] = [tx, ty, tz]

            # Convert to OpenGL convention for transforms.json
            c2w_opengl = P @ c2w_opencv @ P

            poses[frame_id] = c2w_opengl.tolist()

    return poses


def main():
    args = parse_args()

    # Read VGGT-SLAM poses
    vggt_poses = read_vggt_poses(args.poses_file)
    print(f"Read {len(vggt_poses)} poses from {args.poses_file}")
    print(f"  Frame ID range: {min(vggt_poses.keys())} - {max(vggt_poses.keys())}")

    # Read existing transforms.json
    with open(args.transforms_json, 'r') as f:
        transforms = json.load(f)

    print(f"Existing transforms.json has {len(transforms['frames'])} frames")

    # Match frames to VGGT-SLAM poses
    matched = 0
    unmatched = 0
    new_frames = []

    for frame in transforms['frames']:
        # Extract frame index from file_path (e.g., "rgb/0.png" -> 0)
        file_path = frame['file_path']
        basename = os.path.splitext(os.path.basename(file_path))[0]
        try:
            frame_idx = int(basename)
        except ValueError:
            # Try to extract number from filename
            import re
            match = re.search(r'(\d+)', basename)
            if match:
                frame_idx = int(match.group(1))
            else:
                print(f"  WARNING: Cannot extract frame ID from {file_path}, skipping")
                unmatched += 1
                continue

        if frame_idx in vggt_poses:
            # Replace the pose with VGGT-SLAM's estimated pose
            frame['transform_matrix'] = vggt_poses[frame_idx]
            new_frames.append(frame)
            matched += 1
        else:
            # This frame doesn't have a VGGT-SLAM pose (wasn't selected as keyframe)
            # Skip it — only include frames with known poses
            unmatched += 1

    print(f"  Matched: {matched} frames")
    print(f"  Unmatched (skipped): {unmatched} frames")

    # Update transforms with only matched frames
    transforms['frames'] = new_frames

    # Determine output path
    if args.output is None:
        base_dir = os.path.dirname(args.transforms_json)
        args.output = os.path.join(base_dir, "transforms_vggt.json")

    # Save
    with open(args.output, 'w') as f:
        json.dump(transforms, f, indent=4)

    print(f"\nSaved to: {args.output}")
    print(f"  {len(new_frames)} frames with VGGT-SLAM poses")
    print(f"\nTo use with SplaTAM's Gaussian Splatting:")
    print(f"  1. Backup:  cp transforms.json transforms_arkit.json")
    print(f"  2. Replace: cp {args.output} {os.path.join(os.path.dirname(args.transforms_json), 'transforms.json')}")
    print(f"  3. Update config num_frames = {len(new_frames)}")
    print(f"  4. Run:     python scripts/gaussian_splatting.py configs/iphone/gaussian_splatting.py")


if __name__ == "__main__":
    main()
