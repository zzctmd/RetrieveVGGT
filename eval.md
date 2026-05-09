# Evaluation

## Datasets

### 7-Scenes

Download from the [official website](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) or use the [SimpleRecon script](https://github.com/nianticlabs/simplerecon/blob/477aa5b32aa1b93f53abc72828f86023b6e46ce7/data_scripts/7scenes_preprocessing.py#L43):

```bash
git clone https://github.com/nianticlabs/simplerecon.git /tmp/simplerecon
cd /tmp/simplerecon
python data_scripts/7scenes_preprocessing.py --data_dir /path/to/7scenes
```

Expected structure:

```
/path/to/7scenes/
├── chess/
│   ├── TestSplit.txt
│   ├── TrainSplit.txt
│   └── seq-01/
│       ├── frame-000000.color.png
│       ├── frame-000000.depth.png
│       ├── frame-000000.pose.txt
│       └── ...
├── fire/
├── heads/
├── office/
├── pumpkin/
├── redkitchen/
└── stairs/
```

Create symlink (recommended):

```bash
mkdir -p src/data
ln -s /path/to/7scenes src/data/7scenes
```

### Neural RGBD

Download:
```bash
bash data/download_nrgbd.sh /path/to/download
```

Preprocessing:
```bash
ln -s /path/to/download/neural_rgbd_data src/data/nrgbd
```

### Bonn RGB-D Dynamic

Download:
```bash
bash data/download_bonn.sh /path/to/download
```

Preprocessing:
```bash
python datasets_preprocess/prepare_bonn.py \
    --data_dir /path/to/download/bonn/rgbd_bonn_dataset \
    --output_dir src/data/long_bonn_s1/rgbd_bonn_dataset \
    --frames 50,200,300,400,500
```

### TUM freiburg3

Download:
```bash
bash data/download_tum_dynamics.sh /path/to/download
```

Preprocessing:
```bash
python datasets_preprocess/prepare_tum_dynamic.py \
    --data_dir /path/to/download/tum \
    --output_dir src/data/long_tum_dynamic_s1 \
    --frames 50,200,500,1000
```

---

## Evaluation Scripts

All evaluation scripts are in `scripts/`. Each supports environment-variable overrides:

| Task | Script | Example |
|------|--------|---------|
| 3D Reconstruction | `scripts/run_recon.sh` | `GPU_ID=0 DATASET=nrgbd FRAMES="200 500" bash scripts/run_recon.sh` |
| Video Depth | `scripts/run_depth.sh` | `GPU_ID=0 DATASETS="bonn_s1_200 bonn_s1_500" bash scripts/run_depth.sh` |
| Camera Pose | `scripts/run_pose.sh` | `GPU_ID=0 DATASETS="tum_dynamic_s1_200 tum_dynamic_s1_500" bash scripts/run_pose.sh` |

### Quick Start: 7-Scenes 500-Frame

```bash
cd src/
CUDA_VISIBLE_DEVICES=0 python eval/mv_recon/launch_retrieve.py \
    --weights ../ckpt/checkpoints.pth \
    --dataset 7scenes \
    --size 518 \
    --output_dir ../eval_results/7scenes_500 \
    --max_frames 500 \
    --top_k 47 \
    --anchor 1 \
    --use_segment_sampling \
    --segment_threshold_mode "mean+0.3std"
```

Or simply: `bash scripts/run_recon.sh`

Results are saved in `eval_results/7scenes_500/7scenes/`. Average metrics are printed at the end:

```
avg Acc: X.XX, avg Comp: X.XX, avg NC1: X.XX, avg NC2: X.XX
```

### StreamVGGT Baseline (without retrieval)

```bash
cd src/ && bash eval/mv_recon/run.sh
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--top_k` | 47 | Number of query-relevant historical frames to retrieve |
| `--anchor` | 1 | Number of anchor frames (always kept) |
| `--use_segment_sampling` | flag | Enable segment sampling for diverse coverage |
| `--segment_threshold_mode` | `mean+0.3std` | Similarity threshold for segment detection |
| `--max_frames` | 500 | Max frames to process per sequence |
| `--size` | 518 | Input resolution (518×392) |
| `--use_pose_aware` | flag | Enable Pose-Aware spatial region classification |
| `--use_kv_compression` | flag | Enable per-region KV compression |

**Total context window** = `top_k` + `anchor` = 47 + 1 = **48 frames** per step.
