# Datasets

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

### NRGBD

Download:
```bash
bash data/download_nrgbd.sh /path/to/download
```

Preprocessing:
```bash
ln -s /path/to/download/neural_rgbd_data src/data/nrgbd
```

### Bonn

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

### TUM Dynamic

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
