"""
Prepare TUM RGB-D Dynamic dataset for long-sequence evaluation.

Associates RGB images with groundtruth poses via timestamps,
then extracts specified frame counts at a given sampling interval.

Usage:
    python datasets_preprocess/prepare_tum_dynamic.py \
        --data_dir /path/to/tum \
        --output_dir src/data/long_tum_dynamic_s1
"""

import argparse
import glob
import os
import shutil


def read_file_list(filename):
    """Read a TUM-format trajectory/rgb file: 'timestamp d1 d2 d3 ...'"""
    with open(filename) as f:
        data = f.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    entries = [
        [v.strip() for v in line.split(" ") if v.strip() != ""]
        for line in lines
        if len(line) > 0 and line[0] != "#"
    ]
    return dict((float(l[0]), l[1:]) for l in entries if len(l) > 1)


def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    """Associate two dictionaries by closest timestamps."""
    first_keys = set(first_list.keys())
    second_keys = set(second_list.keys())

    potential_matches = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]
    potential_matches.sort()

    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches


def prepare_tum_dynamic(data_dir, output_dir, sample_interval=1, frame_list=None):
    if frame_list is None:
        frame_list = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    os.makedirs(output_dir, exist_ok=True)

    dirs = sorted(glob.glob(os.path.join(data_dir, "*/")))
    if not dirs:
        print(f"Error: No subdirectories found in {data_dir}")
        return

    for target_frames in frame_list:
        print(f"\n=== Processing {target_frames} frames (interval={sample_interval}) ===")

        for d in dirs:
            rgb_file = os.path.join(d, "rgb.txt")
            gt_file = os.path.join(d, "groundtruth.txt")

            if not os.path.exists(rgb_file) or not os.path.exists(gt_file):
                continue

            first_list = read_file_list(rgb_file)
            second_list = read_file_list(gt_file)
            matches = associate(first_list, second_list, 0.0, 0.02)

            frames = []
            gt = []
            for a, b in matches:
                frames.append(os.path.join(d, first_list[a][0]))
                gt.append([b] + second_list[b])

            # Sample and truncate
            frames = frames[::sample_interval][:target_frames]
            gt_sampled = gt[::sample_interval][:target_frames]

            dir_name = os.path.basename(os.path.dirname(d))
            rgb_out = os.path.join(output_dir, dir_name, f"rgb_{target_frames}")
            os.makedirs(rgb_out, exist_ok=True)

            for frame in frames:
                shutil.copy(frame, rgb_out)

            gt_out = os.path.join(output_dir, dir_name, f"groundtruth_{target_frames}.txt")
            with open(gt_out, "w") as f:
                for pose in gt_sampled:
                    f.write(f"{' '.join(map(str, pose))}\n")

            print(f"  {dir_name}: {len(frames)} frames")

    print(f"\nDone! Output saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare TUM Dynamic dataset for long-sequence evaluation")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to tum/ directory containing freiburg3 sequences")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory (e.g., src/data/long_tum_dynamic_s1)")
    parser.add_argument("--sample_interval", type=int, default=1,
                        help="Sampling interval (default: 1, i.e., every frame)")
    parser.add_argument("--frames", type=str, default="50,200,500,1000",
                        help="Comma-separated list of target frame counts")
    args = parser.parse_args()

    frame_list = [int(x) for x in args.frames.split(",")]
    prepare_tum_dynamic(args.data_dir, args.output_dir, args.sample_interval, frame_list)
