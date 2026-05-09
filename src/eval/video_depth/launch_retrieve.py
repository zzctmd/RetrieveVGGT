"""
RetrieveVGGT Video Depth Evaluation Script
with Query-Driven frame selection.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import numpy as np
import torch
import argparse

from eval.video_depth.metadata import dataset_metadata
from eval.video_depth.utils import save_depth_maps
from accelerate import PartialState
import time
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser("RetrieveVGGT Video Depth evaluation", add_help=False)
    parser.add_argument("--weights", type=str, default="", help="ckpt path")
    parser.add_argument("--output_dir", type=str, default="", help="output directory")
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--eval_dataset", type=str, default="bonn_s1_50",
                       choices=list(dataset_metadata.keys()))
    parser.add_argument("--pose_eval_stride", default=1, type=int, help="stride for evaluation")
    parser.add_argument("--full_seq", action="store_true", default=False,
                       help="use full sequence for evaluation")
    parser.add_argument("--seq_list", nargs="+", default=None,
                       help="list of sequences for evaluation")
    
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


def prepare_input(img_paths, size, crop=True):
    """Convert image list to model input views."""
    images = load_images(img_paths, size=size, crop=crop)
    views = []
    for i in range(len(images)):
        view = {
            "img": images[i]["img"].to(device='cuda'),
            "ray_map": torch.full(
                (
                    images[i]["img"].shape[0],
                    6,
                    images[i]["img"].shape[-2],
                    images[i]["img"].shape[-1],
                ),
                torch.nan,
            ).to(device='cuda'),
            "true_shape": torch.from_numpy(images[i]["true_shape"]).to(device='cuda'),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(
                np.eye(4).astype(np.float32)
            ).unsqueeze(0).to(device='cuda'),
            "img_mask": torch.tensor(True).unsqueeze(0).to(device='cuda'),
            "ray_mask": torch.tensor(False).unsqueeze(0).to(device='cuda'),
            "update": torch.tensor(True).unsqueeze(0).to(device='cuda'),
            "reset": torch.tensor(False).unsqueeze(0).to(device='cuda'),
        }
        views.append(view)
    return views


def prepare_output(outputs):
    """Extract depth and confidence from model output."""
    pts3ds_self = [output["depth"].cpu() for output in outputs["pred"]]
    conf_self = [output["depth_conf"].cpu() for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self, 0)
    return pts3ds_self, conf_self


def eval_video_depth(args, model, save_dir=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]

    eval_video_depth_dist(args, model, save_dir=save_dir, img_path=img_path)


def eval_video_depth_dist(args, model, img_path, save_dir=None):
    from dust3r.inference import loss_of_one_batch

    metadata = dataset_metadata.get(args.eval_dataset)

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

    # Compute tokens_per_frame and enable Query-Driven mode
    # Infer from first sequence (all sequences share the same resolution)
    H, W = _get_resolution(args.size)
    tokens_per_frame = 5 + (H // 14) * (W // 14)

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

                views = prepare_input(filelist, size=args.size, crop=False)
                for view in views:
                    view["img"] = (view["img"] + 1.0) / 2.0

                # Reset KV repository before new sequence
                model.aggregator.reset_kv_repository()
                
                start = time.time()
                outputs = loss_of_one_batch(views, model, None, None, inference=True)
                end = time.time()
                
                fps = len(filelist) / (end - start)
                print(f"[RetrieveVGGT] {args.eval_dataset} {seq: <16}, FPS: {fps:.2f}")
                
                with torch.cuda.amp.autocast(dtype=torch.float32):
                    pts3ds_self, conf_self = prepare_output(outputs)
                    os.makedirs(f"{save_dir}/{seq}", exist_ok=True)
                    save_depth_maps(pts3ds_self, f"{save_dir}/{seq}", conf_self=conf_self)
                
                # Cleanup memory
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
                    print(f"Depth evaluation error in sequence {seq}, skipping.")
                else:
                    raise e

    model.aggregator.disable_query_driven()
    print("[RetrieveVGGT] Evaluation complete.")


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
    parser = get_args_parser()
    args = parser.parse_args()
    
    from dust3r.utils.image import load_images_for_eval as load_images
    from streamvggt.models.streamvggt import StreamVGGT

    if args.eval_dataset == "sintel":
        args.full_seq = True
    else:
        args.full_seq = False

    print("[RetrieveVGGT] Loading model...")
    model = StreamVGGT()
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    model = model.to("cuda")
    del ckpt
    print("[RetrieveVGGT] Model loaded successfully.")
    
    with torch.no_grad():
        eval_video_depth(args, model, save_dir=args.output_dir)
