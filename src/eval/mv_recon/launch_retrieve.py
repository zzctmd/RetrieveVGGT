"""
RetrieveVGGT 3D Reconstruction Evaluation Script
Supports 7-Scenes / Neural RGBD datasets with Query-Driven frame selection.
"""
import os
import sys
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from accelerate import Accelerator
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from collections import defaultdict
import re


def get_args_parser():
    parser = argparse.ArgumentParser("RetrieveVGGT 3D Reconstruction evaluation", add_help=False)
    parser.add_argument("--weights", type=str, default="", help="ckpt path")
    parser.add_argument("--output_dir", type=str, default="", help="output directory")
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--max_frames", type=int, default=500, help="max frames limit")
    parser.add_argument("--dataset", type=str, default="7scenes",
                       choices=["7scenes", "nrgbd"],
                       help="evaluation dataset")
    parser.add_argument("--data_dir", type=str, default="",
                       help="dataset root (default: data/7scenes or data/nrgbd)")
    
    # RetrieveVGGT Query-Driven parameters
    parser.add_argument("--top_k", type=int, default=47, help="number of top-k frames to retrieve")
    parser.add_argument("--anchor", type=int, default=1, help="number of anchor frames")
    parser.add_argument("--use_segment_sampling", action="store_true",
                       help="enable Segment Sampling for sparse diverse selection")
    parser.add_argument("--segment_threshold_mode", type=str, default='mean',
                       choices=['mean', 'mean+0.3std', 'mean+0.6std'],
                       help="similarity threshold mode for Segment Sampling")
    
    # Pose-Aware spatial region classification
    parser.add_argument("--use_pose_aware", action="store_true",
                       help="enable Pose-Aware spatial region classification")
    parser.add_argument("--pos_cluster_radius", type=float, default=0,
                       help="position grid size (0=auto BBox grid)")

    parser.add_argument("--grid_k", type=int, default=3,
                       help="grid resolution K (K cells per axis)")
    parser.add_argument("--n_dir_bins", type=int, default=4,
                       help="number of azimuth direction bins D")
    
    # KV Compression
    parser.add_argument("--use_kv_compression", action="store_true",
                       help="enable periodic KV compression")
    parser.add_argument("--compression_interval", type=int, default=200,
                       help="compress every N frames")
    parser.add_argument("--compression_region_cap_ratio", type=float, default=1.0,
                       help="region capacity cap ratio")
    parser.add_argument("--compression_delete_ratio", type=float, default=0.5,
                       help="deletion ratio for over-cap regions")
    parser.add_argument("--compression_min_kv_ratio", type=float, default=0.2,
                       help="minimum KV retention ratio")
    
    return parser


pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""
regex = re.compile(pattern, re.VERBOSE)


def build_dataset(args, resolution):
    """Build dataset based on --dataset argument."""
    if args.dataset == "7scenes":
        from eval.mv_recon.data import SevenScenes
        data_dir = args.data_dir or "data/7scenes"
        return {
            "7scenes": SevenScenes(
                split="test",
                ROOT=data_dir,
                resolution=resolution,
                num_seq=1,
                full_video=True,
                kf_every=2,
                max_frames=args.max_frames,
            ),
        }
    elif args.dataset == "nrgbd":
        from eval.mv_recon.data import NRGBD
        data_dir = args.data_dir or "data/nrgbd"
        return {
            "nrgbd": NRGBD(
                split="test",
                ROOT=data_dir,
                resolution=resolution,
                num_seq=1,
                full_video=True,
                kf_every=2,
                max_frames=args.max_frames,
            ),
        }
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def main(args):
    from eval.mv_recon.utils import accuracy, completion
    from streamvggt.models.streamvggt import StreamVGGT
    from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    elif args.size == 518:
        resolution = (518, 392)
    else:
        raise NotImplementedError
    
    datasets_all = build_dataset(args, resolution)

    accelerator = Accelerator()
    device = accelerator.device
    
    # Load model
    print(f"[RetrieveVGGT] Loading model for {args.dataset}...")
    model = StreamVGGT()
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    model = model.to("cuda")
    del ckpt
    
    # Compute tokens_per_frame and enable Query-Driven mode
    H, W = resolution if isinstance(resolution, tuple) else (resolution, resolution)
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
    
    os.makedirs(args.output_dir, exist_ok=True)
    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    with torch.no_grad():
        for name_data, dataset in datasets_all.items():
            save_path = osp.join(args.output_dir, name_data)
            os.makedirs(save_path, exist_ok=True)
            log_file = osp.join(save_path, f"logs_{accelerator.process_index}.txt")

            acc_all = 0
            acc_all_med = 0
            comp_all = 0
            comp_all_med = 0
            nc1_all = 0
            nc1_all_med = 0
            nc2_all = 0
            nc2_all_med = 0

            with accelerator.split_between_processes(list(range(len(dataset)))) as idxs:
                for data_idx in tqdm(idxs):
                    batch = default_collate([dataset[data_idx]])
                    ignore_keys = set(["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"])
                    for view in batch:
                        for name in view.keys():
                            if name in ignore_keys:
                                continue
                            if isinstance(view[name], (tuple, list)):
                                view[name] = [x.to(device, non_blocking=True) for x in view[name]]
                            else:
                                view[name] = view[name].to(device, non_blocking=True)

                    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                    with torch.cuda.amp.autocast(dtype=dtype):
                        for view in batch:
                            view["img"] = (view["img"] + 1.0) / 2.0

                    # Reset KV repository before new sequence
                    model.aggregator.reset_kv_repository()
                    
                    with torch.cuda.amp.autocast(dtype=dtype):
                        results = model.inference(batch)
                        preds, batch = results.ress, results.views

                    # Evaluation
                    scene_id = batch[-1]["label"][0].rsplit("/", 1)[0]
                    print(f"[RetrieveVGGT] Eval {scene_id} ({data_idx+1}/{len(dataset)})")
                    gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                        criterion.get_all_pts3d_t(batch, preds)
                    )

                    pts_all = []
                    pts_gt_all = []
                    images_all = []
                    masks_all = []

                    for j, view in enumerate(batch):
                        image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                        mask = view["valid_mask"].cpu().numpy()[0]
                        pts = pred_pts[j].cpu().numpy()[0]
                        pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                        H, W = image.shape[:2]
                        cx, cy = W // 2, H // 2
                        l, t = cx - 112, cy - 112
                        r, b = cx + 112, cy + 112
                        image = image[t:b, l:r]
                        mask = mask[t:b, l:r]
                        pts = pts[t:b, l:r]
                        pts_gt = pts_gt[t:b, l:r]

                        images_all.append(image[None, ...])
                        pts_all.append(pts[None, ...])
                        pts_gt_all.append(pts_gt[None, ...])
                        masks_all.append(mask[None, ...])

                    images_all = np.concatenate(images_all, axis=0)
                    pts_all = np.concatenate(pts_all, axis=0)
                    pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                    masks_all = np.concatenate(masks_all, axis=0)

                    np.save(
                        os.path.join(save_path, f"{scene_id.replace('/', '_')}.npy"),
                        {"images_all": images_all, "pts_all": pts_all,
                         "pts_gt_all": pts_gt_all, "masks_all": masks_all},
                    )

                    threshold = 0.1

                    pts_all_masked = pts_all[masks_all > 0]
                    pts_gt_all_masked = pts_gt_all[masks_all > 0]
                    images_all_masked = images_all[masks_all > 0]

                    mask = np.isfinite(pts_all_masked)
                    pts_all_masked = pts_all_masked[mask]
                    pts_gt_all_masked = pts_gt_all_masked[mask]

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pts_all_masked.reshape(-1, 3))
                    pcd.colors = o3d.utility.Vector3dVector(images_all_masked.reshape(-1, 3))
                    o3d.io.write_point_cloud(
                        os.path.join(save_path, f"{scene_id.replace('/', '_')}-mask.ply"), pcd)

                    pcd_gt = o3d.geometry.PointCloud()
                    pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all_masked.reshape(-1, 3))
                    pcd_gt.colors = o3d.utility.Vector3dVector(images_all_masked.reshape(-1, 3))
                    o3d.io.write_point_cloud(
                        os.path.join(save_path, f"{scene_id.replace('/', '_')}-gt.ply"), pcd_gt)

                    trans_init = np.eye(4)
                    reg_p2p = o3d.pipelines.registration.registration_icp(
                        pcd, pcd_gt, threshold, trans_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint())
                    pcd = pcd.transform(reg_p2p.transformation)

                    o3d.io.write_point_cloud(
                        os.path.join(save_path, f"{scene_id.replace('/', '_')}-mask_align.ply"), pcd)

                    pcd.estimate_normals()
                    pcd_gt.estimate_normals()

                    gt_normal = np.asarray(pcd_gt.normals)
                    pred_normal = np.asarray(pcd.normals)

                    acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd.points, gt_normal, pred_normal)
                    comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gt_normal, pred_normal)
                    
                    print(
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - "
                        f"Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                    )
                    print(
                        f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - "
                        f"Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}",
                        file=open(log_file, "a"),
                    )

                    acc_all += acc
                    comp_all += comp
                    nc1_all += nc1
                    nc2_all += nc2
                    acc_all_med += acc_med
                    comp_all_med += comp_med
                    nc1_all_med += nc1_med
                    nc2_all_med += nc2_med

                    # Cleanup current sequence
                    model.aggregator.reset_kv_repository()
                    torch.cuda.empty_cache()
                    gc.collect()

            accelerator.wait_for_everyone()
            
            if accelerator.is_main_process:
                to_write = ""
                for i in range(8):
                    if not os.path.exists(osp.join(save_path, f"logs_{i}.txt")):
                        break
                    with open(osp.join(save_path, f"logs_{i}.txt"), "r") as f_sub:
                        to_write += f_sub.read()

                with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
                    log_data = to_write
                    metrics = defaultdict(list)
                    for line in log_data.strip().split("\n"):
                        match = regex.match(line)
                        if match:
                            data = match.groupdict()
                            for key, value in data.items():
                                if key != "scene_id":
                                    metrics[key].append(float(value))
                            metrics["nc"].append((float(data["nc1"]) + float(data["nc2"])) / 2)
                            metrics["nc_med"].append((float(data["nc1_med"]) + float(data["nc2_med"])) / 2)
                    
                    mean_metrics = {
                        metric: sum(values) / len(values)
                        for metric, values in metrics.items()
                    }

                    print_str = f"{'mean'.ljust(20)}: "
                    for m_name in mean_metrics:
                        print_str += f"{m_name}: {np.mean(mean_metrics[m_name]):.3f} | "
                    print_str += "\n"
                    f.write(to_write + print_str)
    
    model.aggregator.disable_query_driven()
    print("[RetrieveVGGT] Evaluation complete.")


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
