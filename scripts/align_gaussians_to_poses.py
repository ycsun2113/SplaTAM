"""
Align a SplaTAM Gaussian map to a new set of camera poses.

Given:
  - params.npz from a SplaTAM run (Gaussian map + estimated poses)
  - transforms.json with better poses (e.g., from COLMAP, ARKit, VGGT)

This script:
  1. Extracts SplaTAM's estimated camera positions from params.npz
  2. Extracts the target poses from transforms.json
  3. Computes a Umeyama (similarity) alignment: rotation + translation + scale
  4. Applies the transform to all Gaussian positions and rotations
  5. Replaces the camera poses with the target poses
  6. Saves a new params.npz

Usage:
  python scripts/align_gaussians_to_poses.py \
      --params ./experiments/iphone/scan_01/SplaTAM_R3D/params.npz \
      --poses /mnt/ws-frb/users/ycs/datasets/splatam/scan_01/transforms.json \
      --output ./experiments/iphone/scan_01/SplaTAM_R3D/params_aligned.npz
"""

import argparse
import json
import sys

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Align SplaTAM Gaussians to new poses")
    parser.add_argument("--params", required=True, help="Path to SplaTAM params.npz")
    parser.add_argument("--poses", required=True, help="Path to transforms.json with target poses")
    parser.add_argument("--output", required=True, help="Output path for aligned params.npz")
    parser.add_argument("--num_frames", type=int, default=-1,
                        help="Number of frames to use for alignment (-1 = all)")
    return parser.parse_args()


def extract_splatam_camera_positions(params):
    """Extract camera positions from SplaTAM's estimated poses in params.npz."""
    cam_rots_quat = torch.tensor(params['cam_unnorm_rots']).float()  # (1, 4, N)
    cam_trans = torch.tensor(params['cam_trans']).float()             # (1, 3, N)
    num_frames = cam_trans.shape[-1]

    positions = []
    for i in range(num_frames):
        # Get w2c for frame i
        quat = torch.nn.functional.normalize(cam_rots_quat[..., i])  # (1, 4)
        t = cam_trans[..., i]  # (1, 3)
        # Build rotation matrix from quaternion
        R = quaternion_to_rotation_matrix(quat.squeeze())  # (3, 3)
        # w2c: [R | t], camera position in world = -R^T @ t
        cam_pos = -R.T @ t.squeeze()
        positions.append(cam_pos.numpy())

    return np.array(positions)  # (N, 3)


def quaternion_to_rotation_matrix(q):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = torch.zeros(3, 3)
    R[0, 0] = 1 - 2*(y*y + z*z)
    R[0, 1] = 2*(x*y - w*z)
    R[0, 2] = 2*(x*z + w*y)
    R[1, 0] = 2*(x*y + w*z)
    R[1, 1] = 1 - 2*(x*x + z*z)
    R[1, 2] = 2*(y*z - w*x)
    R[2, 0] = 2*(x*z - w*y)
    R[2, 1] = 2*(y*z + w*x)
    R[2, 2] = 1 - 2*(x*x + y*y)
    return R


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def extract_target_positions(transforms_path):
    """Extract camera positions from transforms.json (c2w matrices)."""
    with open(transforms_path) as f:
        data = json.load(f)

    positions = []
    w2c_list = []
    for frame in data['frames']:
        c2w = np.array(frame['transform_matrix'], dtype=np.float64)
        # Apply the same coordinate flip as NeRFCaptureDataset
        P = np.array([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]], dtype=np.float64)
        c2w = P @ c2w @ P.T
        w2c = np.linalg.inv(c2w)
        # Make poses relative to first frame
        w2c_list.append(w2c)
        # Camera position in world frame
        cam_pos = c2w[:3, 3]
        positions.append(cam_pos)

    # Make relative to first frame
    first_w2c = w2c_list[0]
    rel_positions = []
    rel_w2c_list = []
    for w2c in w2c_list:
        rel_w2c = w2c @ np.linalg.inv(first_w2c)
        rel_w2c_list.append(rel_w2c)
        rel_c2w = np.linalg.inv(rel_w2c)
        rel_positions.append(rel_c2w[:3, 3])

    return np.array(rel_positions), rel_w2c_list


def umeyama_alignment(src, dst):
    """Compute similarity transform (s*R*src + t = dst) using Umeyama method.

    Args:
        src: (N, 3) source points
        dst: (N, 3) destination points

    Returns:
        s: scale factor
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    assert src.shape == dst.shape
    n, d = src.shape

    # Centroids
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    # Centered
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # Covariance
    sigma = (dst_c.T @ src_c) / n

    U, D, Vt = np.linalg.svd(sigma)

    # Handle reflection
    S = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[d - 1, d - 1] = -1

    R = U @ S @ Vt

    # Scale
    var_src = np.sum(src_c ** 2) / n
    s = np.trace(np.diag(D) @ S) / var_src

    # Translation
    t = mu_dst - s * R @ mu_src

    return s, R, t


def apply_transform_to_gaussians(params, s, R, t):
    """Apply similarity transform to Gaussian positions and rotations."""
    # Transform means3D: new_pos = s * R @ old_pos + t
    means3D = params['means3D'].copy()
    means3D = (s * (R @ means3D.T).T + t).astype(np.float32)
    params['means3D'] = means3D

    # Transform rotations
    R_torch = torch.tensor(R, dtype=torch.float32)
    unnorm_rots = torch.tensor(params['unnorm_rotations'], dtype=torch.float32)
    num_pts = unnorm_rots.shape[0]
    for i in range(num_pts):
        q = unnorm_rots[i]
        q_norm = torch.nn.functional.normalize(q.unsqueeze(0)).squeeze()
        R_gauss = quaternion_to_rotation_matrix(q_norm)
        R_new = R_torch @ R_gauss
        q_new = rotation_matrix_to_quaternion(R_new.numpy())
        unnorm_rots[i] = torch.tensor(q_new, dtype=torch.float32)
    params['unnorm_rotations'] = unnorm_rots.numpy()

    # Scale the log_scales
    log_scales = params['log_scales'].copy()
    log_scales += np.log(s)
    params['log_scales'] = log_scales.astype(np.float32)

    return params


def replace_camera_poses(params, rel_w2c_list, num_frames):
    """Replace SplaTAM camera poses with target poses."""
    cam_rots = np.zeros((1, 4, num_frames), dtype=np.float32)
    cam_trans = np.zeros((1, 3, num_frames), dtype=np.float32)
    # Initialize all to identity quaternion
    cam_rots[0, 0, :] = 1.0  # w=1, x=y=z=0

    for i, w2c in enumerate(rel_w2c_list[:num_frames]):
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        q = rotation_matrix_to_quaternion(R)
        cam_rots[0, :, i] = q
        cam_trans[0, :, i] = t

    params['cam_unnorm_rots'] = cam_rots.astype(np.float32)
    params['cam_trans'] = cam_trans.astype(np.float32)
    return params


def main():
    args = parse_args()

    print(f"Loading SplaTAM params from {args.params}")
    params = dict(np.load(args.params, allow_pickle=True))

    print(f"Loading target poses from {args.poses}")
    target_positions, target_w2c_list = extract_target_positions(args.poses)

    num_frames = params['cam_trans'].shape[-1]
    num_target = len(target_positions)
    print(f"SplaTAM frames: {num_frames}, Target frames: {num_target}")

    n_align = min(num_frames, num_target)
    if args.num_frames > 0:
        n_align = min(n_align, args.num_frames)

    print(f"Using {n_align} frames for alignment")

    # Extract SplaTAM camera positions
    splatam_positions = extract_splatam_camera_positions(params)[:n_align]
    target_pos_subset = target_positions[:n_align]

    # Compute alignment
    s, R, t = umeyama_alignment(splatam_positions, target_pos_subset)
    print(f"Alignment: scale={s:.4f}")
    print(f"  Rotation:\n{R}")
    print(f"  Translation: {t}")

    # Compute alignment error
    aligned = s * (R @ splatam_positions.T).T + t
    error = np.sqrt(np.mean(np.sum((aligned - target_pos_subset) ** 2, axis=1)))
    print(f"  RMSE after alignment: {error:.4f}m")

    # Apply transform to Gaussians
    print("Transforming Gaussian map...")
    params = apply_transform_to_gaussians(params, s, R, t)

    # Replace camera poses
    print("Replacing camera poses...")
    params = replace_camera_poses(params, target_w2c_list, num_frames)

    # Save
    print(f"Saving aligned params to {args.output}")
    np.savez(args.output, **params)
    print("Done!")


if __name__ == "__main__":
    main()
