"""
RetrieveVGGT Pose Evaluation Script
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
import torch
import argparse

from eval.pose_evaluation.metadata import dataset_metadata
from eval.pose_evaluation.utils import *

from accelerate import PartialState
from streamvggt.models.streamvggt import StreamVGGT

from tqdm import tqdm
import time

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        help="path to the model weights",
        default="",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )

    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="sintel",
        choices=list(dataset_metadata.keys()),
    )
    parser.add_argument("--size", type=int, default=512)

    parser.add_argument(
        "--pose_eval_stride", default=1, type=int, help="stride for pose evaluation"
    )
    parser.add_argument(
        "--full_seq",
        action="store_true",
        default=False,
        help="use full sequence for pose evaluation",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="list of sequences for pose evaluation",
    )

    parser.add_argument("--solve_pose", action="store_true", default=False)
    
    # RetrieveVGGT Query-Driven parameters
    parser.add_argument("--top_k", type=int, default=46, help="number of top-k frames to retrieve")
    parser.add_argument("--anchor", type=int, default=1, help="number of anchor frames")
    parser.add_argument("--use_segment_sampling", action="store_true",
                       help="enable Segment Sampling")
    parser.add_argument("--segment_threshold_mode", type=str, default='mean',
                       choices=['mean', 'mean+0.3std', 'mean+0.6std'],
                       help="similarity threshold mode")
    
    # Pose-Aware spatial region classification
    parser.add_argument("--use_pose_aware", action="store_true",
                       help="enable Pose-Aware classification")
    parser.add_argument("--pos_cluster_radius", type=float, default=0,
                       help="position grid size (0=auto)")

    parser.add_argument("--grid_k", type=int, default=3,
                       help="grid resolution K")
    parser.add_argument("--n_dir_bins", type=int, default=4,
                       help="azimuth direction bins D")
    
    # KV Compression
    parser.add_argument("--use_kv_compression", action="store_true",
                       help="enable KV compression")
    parser.add_argument("--compression_interval", type=int, default=200,
                       help="compress every N frames")
    parser.add_argument("--compression_region_cap_ratio", type=float, default=1.0,
                       help="region capacity cap ratio")
    parser.add_argument("--compression_delete_ratio", type=float, default=0.5,
                       help="deletion ratio for over-cap regions")
    parser.add_argument("--compression_min_kv_ratio", type=float, default=0.2,
                       help="minimum KV retention ratio")
    
    return parser


def eval_pose_estimation(args, model, save_dir=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]
    mask_path = metadata["mask_path"]

    ate_mean, rpe_trans_mean, rpe_rot_mean = eval_pose_estimation_dist(
        args, model, save_dir=save_dir, img_path=img_path, mask_path=mask_path
    )
    return ate_mean, rpe_trans_mean, rpe_rot_mean


def eval_pose_estimation_dist(args, model, img_path, save_dir=None, mask_path=None):

    metadata = dataset_metadata.get(args.eval_dataset)
    anno_path = metadata.get("anno_path", None)

    seq_list = args.seq_list
    if seq_list is None:
        if metadata.get("full_seq", False):
            args.full_seq = True
        else:
            seq_list = metadata.get("seq_list", [])
        if args.full_seq:
            seq_list = os.listdir(img_path)
            seq_list = [
                seq for seq in seq_list if os.path.isdir(os.path.join(img_path, seq))
            ]
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir

    distributed_state = PartialState()
    model.to(distributed_state.device)
    device = distributed_state.device

    # Enable Query-Driven mode (once before the loop)
    # Infer tokens_per_frame from args.size
    H_res, W_res = _get_resolution(args.size)
    tokens_per_frame = 5 + (H_res // 14) * (W_res // 14)

    model.aggregator.enable_query_driven(
        top_k_frames=args.top_k,
        anchor_frames=args.anchor,
        tokens_per_frame=tokens_per_frame,
        use_segment_sampling=args.use_segment_sampling,
        segment_threshold_mode=args.segment_threshold_mode,
        # Pose-Aware
        use_pose_aware=args.use_pose_aware,
        pos_cluster_radius=args.pos_cluster_radius,
        grid_k=args.grid_k,
        n_dir_bins=args.n_dir_bins,
        # KV Compression
        use_kv_compression=args.use_kv_compression,
        compression_interval=args.compression_interval,
        compression_region_cap_ratio=args.compression_region_cap_ratio,
        compression_delete_ratio=args.compression_delete_ratio,
        compression_min_kv_ratio=args.compression_min_kv_ratio,
    )
    print(f"[RetrieveVGGT] Query-Driven enabled: top_k={args.top_k}, anchor={args.anchor}")

    os.makedirs(save_dir, exist_ok=True)

    with distributed_state.split_between_processes(seq_list) as seqs:
        ate_list = []
        rpe_trans_list = []
        rpe_rot_list = []
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"
        
        for seq in tqdm(seqs):
            try:
                dir_path = metadata["dir_path_func"](img_path, seq)

                # Handle skip_condition
                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(save_dir, seq):
                    continue

                filelist = sorted([
                    os.path.join(dir_path, name) for name in os.listdir(dir_path)
                ])
                filelist = filelist[:: args.pose_eval_stride]

                images = load_and_preprocess_images(filelist).to(device)
                frames = []
                for i in range(images.shape[0]):
                    frame = {"img": images[i].unsqueeze(0)}
                    frames.append(frame)

                # Reset KV repository before new sequence
                model.aggregator.reset_kv_repository()

                start = time.time()
                predictions = {}
                with torch.no_grad():
                    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                    with torch.cuda.amp.autocast(dtype=dtype):
                        output = model.inference(frames)
                end = time.time()
                
                fps = len(filelist) / (end - start)
                print(f"[RetrieveVGGT] {args.eval_dataset} {seq: <16}, FPS: {fps:.2f}")

                all_camera_pose = []
                for res in output.ress:
                    all_camera_pose.append(res['camera_pose'].squeeze(0))
                    
                predictions["pose_enc"] = torch.stack(all_camera_pose, dim=0)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(
                        predictions["pose_enc"].unsqueeze(0) if predictions["pose_enc"].ndim == 2 else predictions["pose_enc"], 
                        images.shape[-2:]
                    )
                predictions["extrinsic"] = extrinsic.squeeze(0)
                predictions["intrinsic"] = intrinsic.squeeze(0) if intrinsic is not None else None

                # Convert (S, 3, 4) extrinsics to (S, 4, 4) SE(3) matrices
                extrinsic = predictions["extrinsic"].to(device)
                add_row = torch.tensor([0, 0, 0, 1], device=device, dtype=extrinsic.dtype).expand(extrinsic.size(0), 1, 4)
                pr_poses = torch.cat((extrinsic, add_row), dim=1)

                # Extract focal length and principal point from intrinsics for saving
                if predictions["intrinsic"] is not None:
                    focals_x = predictions["intrinsic"][:, 0, 0]
                    focals_y = predictions["intrinsic"][:, 1, 1]
                    focal = (focals_x + focals_y) / 2.0
                    pp = predictions["intrinsic"][:, :2, 2]
                    cam_dict = {
                        "focal": focal.cpu().numpy(),
                        "pp": pp.cpu().numpy(),
                    }
                else:
                    H, W = images.shape[-2:]
                    cam_dict = {
                        "focal": np.full(len(images), max(H, W)),
                        "pp": np.tile([W/2, H/2], (len(images), 1)),
                    }

                pred_traj = get_tum_poses(pr_poses)
                os.makedirs(f"{save_dir}/{seq}", exist_ok=True)
                save_tum_poses(pr_poses, f"{save_dir}/{seq}/pred_traj.txt")
                save_focals(cam_dict, f"{save_dir}/{seq}/pred_focal.txt")
                pose_save_path = os.path.join(save_dir, f"{seq}_poses.npz")
                np.savez(
                    pose_save_path,
                    pose_enc=predictions["pose_enc"].cpu().numpy(),
                    extrinsic=predictions["extrinsic"].cpu().numpy()
                )

                print(f"Pose encoding and extrinsics saved to: {pose_save_path}")

                gt_traj_file = metadata["gt_traj_func"](img_path, anno_path, seq)
                traj_format = metadata.get("traj_format", None)

                if args.eval_dataset == "sintel":
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file, stride=args.pose_eval_stride
                    )
                elif traj_format is not None:
                    gt_traj = load_traj(
                        gt_traj_file=gt_traj_file,
                        traj_format=traj_format,
                        stride=args.pose_eval_stride,
                    )
                else:
                    gt_traj = None

                if gt_traj is not None:
                    ate, rpe_trans, rpe_rot = eval_metrics(
                        pred_traj,
                        gt_traj,
                        seq=seq,
                        filename=f"{save_dir}/{seq}_eval_metric.txt",
                    )
                    plot_trajectory(
                        pred_traj, gt_traj, title=seq, filename=f"{save_dir}/{seq}.png"
                    )
                else:
                    ate, rpe_trans, rpe_rot = 0, 0, 0

                ate_list.append(ate)
                rpe_trans_list.append(rpe_trans)
                rpe_rot_list.append(rpe_rot)

                # Write to error log after each sequence
                with open(error_log_path, "a") as f:
                    f.write(
                        f"{args.eval_dataset}-{seq: <16} | ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}\n"
                    )
                    f.write(f"{ate:.5f}\n")
                    f.write(f"{rpe_trans:.5f}\n")
                    f.write(f"{rpe_rot:.5f}\n")
                
                # Cleanup current sequence
                model.aggregator.reset_kv_repository()
                torch.cuda.empty_cache()

            except Exception as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    with open(error_log_path, "a") as f:
                        f.write(
                            f"OOM error in sequence {seq}, skipping this sequence.\n"
                        )
                    print(f"OOM error in sequence {seq}, skipping...")
                elif "Degenerate covariance rank" in str(
                    e
                ) or "Eigenvalues did not converge" in str(e):
                    with open(error_log_path, "a") as f:
                        f.write(f"Exception in sequence {seq}: {str(e)}\n")
                    print(f"Traj evaluation error in sequence {seq}, skipping.")
                else:
                    raise e

    model.aggregator.disable_query_driven()

    distributed_state.wait_for_everyone()

    results = process_directory(save_dir)
    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)

    # Write the averages to the error log (only on the main process)
    if distributed_state.is_main_process:
        with open(f"{save_dir}/_error_log.txt", "a") as f:
            for i in range(distributed_state.num_processes):
                if not os.path.exists(f"{save_dir}/_error_log_{i}.txt"):
                    break
                with open(f"{save_dir}/_error_log_{i}.txt", "r") as f_sub:
                    f.write(f_sub.read())
            f.write(
                f"Average ATE: {avg_ate:.5f}, Average RPE trans: {avg_rpe_trans:.5f}, Average RPE rot: {avg_rpe_rot:.5f}\n"
            )

    return avg_ate, avg_rpe_trans, avg_rpe_rot


def _get_resolution(size):
    """Return (H, W) based on --size argument."""
    if size == 518:
        return (518, 392)
    elif size == 512:
        return (512, 384)
    elif size == 224:
        return (224, 224)
    else:
        raise NotImplementedError(f"Unsupported size: {size}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    from streamvggt.utils.load_fn import load_and_preprocess_images 
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    args.full_seq = False

    print("[RetrieveVGGT] Loading model...")
    model = StreamVGGT()
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    print("[RetrieveVGGT] Model loaded successfully.")

    eval_pose_estimation(args, model, save_dir=args.output_dir)
