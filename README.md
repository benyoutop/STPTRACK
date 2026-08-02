# STPTrack: A Spatio-Temporal Perception Framework for Visual Object Tracking

## Description

STPTrack is a single-object tracker that uses the target location from the previous frame as a spatial prior and injects selected historical appearance features into current-frame feature learning. The implementation is based on ODTrack and retains `odtrack` as its internal script and configuration identifier.

## Dataset Information

No dataset is distributed with this repository. The standard model is trained with the following public datasets:

- [GOT-10k](http://got-10k.aitestunion.com/)
- [TrackingNet](https://tracking-net.org/)
- [LaSOT](https://cis.temple.edu/lasot/)
- [COCO 2017](https://cocodataset.org/)

Evaluation is reported on GOT-10k, LaSOT, LaSOT Extension, and TrackingNet. Users must download each dataset from its official source and comply with its license and terms of use.

A typical data layout is:

```text
STPTrack/
└── data/
    ├── got10k/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── lasot/
    ├── lasot_extension_subset/
    ├── trackingnet/
    │   ├── TRAIN_0/
    │   ├── ...
    │   ├── TRAIN_11/
    │   └── TEST/
    └── coco/
        ├── annotations/
        └── images/
```

The exact directory names may differ. Dataset paths are stored in `lib/train/admin/local.py` for training and `lib/test/evaluation/local.py` for evaluation.

GOT-10k has a one-shot evaluation rule: results submitted to its test server must be produced by a model trained only on the official GOT-10k training split. Use `baseline256_got.yaml` for that setting.

## Code Information

The main directories and entry points are:

```text
experiments/odtrack/          Model, training, and evaluation configurations
lib/models/odtrack/           ViT backbone and STPTrack model implementation
lib/models/layers/            Candidate elimination and prediction-head layers
lib/train/                    Dataset processing, losses, actors, and training loop
lib/test/tracker/odtrack.py   Online tracking procedure and temporal memory handling
lib/test/evaluation/          Dataset loaders and evaluation utilities
tracking/train.py             Training command-line entry point
tracking/test.py              Testing command-line entry point
tracking/analysis_results.py  Result analysis entry point
tracking/profile_model.py     FLOPs, parameter-count, and speed profiling
```

Important configurations include:

- `baseline.yaml`: ViT-Base model trained on GOT-10k, TrackingNet, LaSOT, and COCO.
- `baseline256_got.yaml`: ViT-Base model trained under the GOT-10k one-shot protocol.
- `baseline_large.yaml`: ViT-Large variant.
- `baseline_sepattn.yaml`: separate-attention variant.

## Requirements

The provided installation script targets the following base environment:

- Linux
- Python 3.8
- NVIDIA GPU with CUDA support
- PyTorch 1.9.0
- torchvision 0.10.0
- CUDA Toolkit 10.2

It also installs PyYAML, easydict, Cython, OpenCV, pandas, tqdm, pycocotools, jpeg4py, TensorBoard, tikzplotlib, THOP, colorama, LMDB, SciPy, Visdom, TensorBoardX, Weights & Biases, and timm.

Create the environment and install the dependencies:

```bash
conda create -n stptrack python=3.8
conda activate stptrack
bash install.sh
```

The versions in `install.sh` reproduce the original development environment. A different CUDA or GPU setup may require a compatible PyTorch build.

## Usage Instructions

### 1. Configure local paths

From the project root, generate local path files:

```bash
python tracking/create_default_local_file.py \
  --workspace_dir . \
  --data_dir ./data \
  --save_dir ./output
```

Then verify and, if necessary, edit:

```text
lib/train/admin/local.py
lib/test/evaluation/local.py
```

Do not rely on machine-specific paths already present in these files when running the project on another computer.

### 2. Prepare pretrained weights

Download the MAE-pretrained ViT-Base checkpoint `mae_pretrain_vit_base.pth` and place it at:

```text
pretrained_networks/mae_pretrain_vit_base.pth
```

The configured filename is controlled by `MODEL.PRETRAIN_FILE` in the selected YAML file.

### 3. Train STPTrack

For the standard ViT-Base experiment on multiple GPUs:

```bash
python tracking/train.py \
  --script odtrack \
  --config baseline \
  --save_dir ./output \
  --mode multiple \
  --nproc_per_node 4 \
  --use_wandb 0
```

For GOT-10k-only training, replace `baseline` with `baseline256_got`. Set `--use_wandb 1` to enable Weights & Biases logging. Checkpoints are written under:

```text
output/checkpoints/train/odtrack/<config_name>/
```

### 4. Test a trained model

The model checkpoint expected by the test code is:

```text
output/checkpoints/train/odtrack/<config_name>/ODTrack_ep<epoch>.pth.tar
```

For example, evaluate epoch 300 on LaSOT:

```bash
python tracking/test.py odtrack baseline \
  --dataset_name lasot \
  --runid 300 \
  --threads 8 \
  --num_gpus 1
```

Evaluate a GOT-10k-only model on the GOT-10k test split:

```bash
python tracking/test.py odtrack baseline256_got \
  --dataset_name got10k_test \
  --runid 100 \
  --threads 8 \
  --num_gpus 1
```

Run the result analysis script after updating its tracker and dataset selections:

```bash
python tracking/analysis_results.py
```

To profile model complexity and speed:

```bash
python tracking/profile_model.py --script odtrack --config baseline
```

## Methodology

STPTrack processes an initial template, the current search region, and temporal context with a ViT encoder.

### Spatio-temporal perception

For frame `T`, the tracker uses the confirmed target center from frame `T-1` to crop a perception region from the current search image. Boundary constraints keep this crop inside the valid image area. In the first frame, where no previous motion information exists, the prior is initialized with a zero tensor. The template, search-region, and perception-region tokens are then jointly encoded.

### Cross-frame feature injection

The module uses candidate-elimination features from frame `T-1`. It compares the representative target feature with search tokens, selects the top-`K` tokens with the highest similarity, and stores them as key historical appearance features. These tokens are concatenated with the current template tokens and injected into subsequent ViT layers. They provide an appearance reference when rapid motion makes the previous-location prior inaccurate.

### Prediction and optimization

The prediction head contains parallel convolutional branches for a classification score map, bounding-box size, and center offset. Training uses focal loss for classification and weighted L1 and GIoU losses for box regression:

```text
L = L_cls + 5 * L_1 + 2 * L_GIoU
```

The model uses a ViT-Base encoder initialized with MAE weights and is optimized with AdamW for 300 epochs. The learning rate is reduced at epoch 240. Each training sample contains three `128 x 128` reference frames and two `256 x 256` search frames. Runtime settings are defined in `experiments/odtrack/`.

## Results

STPTrack achieves the following results on four tracking benchmarks:

| Benchmark | Primary metric | STPTrack result |
|---|---:|---:|
| GOT-10k | AO | 74.1 |
| GOT-10k | SR0.5 / SR0.75 | 85.9 / 71.7 |
| LaSOT | AUC | 69.6 |
| LaSOT Extension | AUC | 48.7 |
| TrackingNet | AUC | 80.8 |

The GOT-10k ablation results show the contribution of each proposed module:

| Variant | AO | SR0.5 | SR0.75 |
|---|---:|---:|---:|
| Baseline | 73.0 | 74.9 | 70.3 |
| Baseline + spatio-temporal perception | 73.7 | 85.9 | 70.6 |
| Baseline + cross-frame feature injection | 73.6 | 85.5 | 71.2 |
| Full STPTrack | 74.1 | 85.9 | 71.7 |

## Citations

Please cite STPTrack as:

> *STPTrack: A Spatio-Temporal Perception Framework for Visual Object Tracking*

Complete BibTeX metadata will be added after publication.

This implementation follows the ODTrack framework. Please also cite:

```bibtex
@inproceedings{zheng2024odtrack,
  title     = {ODTrack: Online Dense Temporal Token Learning for Visual Tracking},
  author    = {Zheng, Yaozong and Zhong, Bineng and Liang, Qihua and Mo, Zhiyi and Zhang, Shengping and Li, Xianxian},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  pages     = {7588--7596},
  year      = {2024}
}
```

## License and Contribution Guidelines

This repository is released under the MIT License. See [LICENSE](LICENSE) for the full terms. Dataset files and pretrained weights are governed by their respective licenses and are not covered by this repository's license.

Contributions are welcome through issues and pull requests. A contribution should:

1. Describe the motivation and affected modules.
2. Avoid unrelated refactoring.
3. Include the configuration and commands required to reproduce the change.
4. Report relevant training or evaluation results when model behavior changes.
5. Avoid committing datasets, checkpoints, generated outputs, credentials, or machine-specific absolute paths.

## Acknowledgments

STPTrack is implemented on top of ODTrack and uses components and conventions derived from OSTrack, STARK, and PyTracking. We thank the authors of these projects and the maintainers of the public tracking benchmarks.
