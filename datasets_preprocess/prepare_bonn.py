"""
Prepare Bonn RGB-D Dynamic dataset for long-sequence evaluation.

Extracts specified frame ranges from each sequence, aligning RGB, depth, and groundtruth.
Output follows the format expected by eval/video_depth and eval/pose_evaluation.

Usage:
    python datasets_preprocess/prepare_bonn.py \
        --data_dir /path/to/bonn/rgbd_bonn_dataset \
        --output_dir src/data/long_bonn_s1/rgbd_bonn_dataset
"""

import argparse
import glob
import os
import shutil
import numpy as np


def prepare_bonn(data_dir, output_dir, start_frame=30, frame_list=None):
    if frame_list is None:
        frame_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    dirs = sorted(glob.glob(os.path.join(data_dir, "*/")))
    if not dirs:
        print(f"Error: No subdirectories found in {data_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    for target_frames in frame_list:
        end_frame = start_frame + target_frames
        print(f"\n=== Processing {target_frames} frames (range: {start_frame}-{end_frame}) ===")

        for d in dirs:
            dir_name = os.path.basename(os.path.dirname(d))
            new_base_dir = os.path.join(output_dir, dir_name)

            # Load all modalities
            rgb_frames = sorted(glob.glob(os.path.join(d, "rgb", "*.png")))
            depth_frames = sorted(glob.glob(os.path.join(d, "depth", "*.png")))
            gt_path = os.path.join(d, "groundtruth.txt")

            if not os.path.exists(gt_path):
                print(f"  Skipping {dir_name}: no groundtruth.txt")
                continue

            gt = np.loadtxt(gt_path)

            # Calculate unified frame count
            actual_counts = [
                min(len(rgb_frames) - start_frame, target_frames),
                min(len(depth_frames) - start_frame, target_frames),
                min(len(gt) - start_frame, target_frames),
            ]
            final_count = min(actual_counts)

            if final_count <= 0:
                print(f"  Skipping {dir_name}: insufficient frames")
                continue

            print(f"  {dir_name}: {final_count} frames")

            # Process RGB
            selected_rgb = rgb_frames[start_frame:start_frame + final_count]
            rgb_dir = os.path.join(new_base_dir, f"rgb_{target_frames}")
            if os.path.exists(rgb_dir):
                shutil.rmtree(rgb_dir)
            os.makedirs(rgb_dir, exist_ok=True)
            for frame in selected_rgb:
                shutil.copy(frame, rgb_dir)

            # Process Depth
            selected_depth = depth_frames[start_frame:start_frame + final_count]
            depth_dir = os.path.join(new_base_dir, f"depth_{target_frames}")
            if os.path.exists(depth_dir):
                shutil.rmtree(depth_dir)
            os.makedirs(depth_dir, exist_ok=True)
            for frame in selected_depth:
                shutil.copy(frame, depth_dir)

            # Process Groundtruth
            gt_final = gt[start_frame:start_frame + final_count]
            gt_file = os.path.join(new_base_dir, f"groundtruth_{target_frames}.txt")
            np.savetxt(gt_file, gt_final)

    print(f"\nDone! Output saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Bonn dataset for long-sequence evaluation")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to bonn/rgbd_bonn_dataset/")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory (e.g., src/data/long_bonn_s1/rgbd_bonn_dataset)")
    parser.add_argument("--start_frame", type=int, default=30,
                        help="Start frame index (default: 30)")
    parser.add_argument("--frames", type=str, default="50,100,150,200,250,300,350,400,450,500",
                        help="Comma-separated list of target frame counts")
    args = parser.parse_args()

    frame_list = [int(x) for x in args.frames.split(",")]
    prepare_bonn(args.data_dir, args.output_dir, args.start_frame, frame_list)
