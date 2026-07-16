# STPTrack：时空感知单目标跟踪算法

STPTrack（Spatio-Temporal Perception Tracking）是一个面向视觉单目标跟踪任务的时空感知跟踪框架。本项目基于 [ODTrack](https://github.com/GXNU-ZhongLab/ODTrack) 开发，内部仍保留 `odtrack` 作为训练与测试脚本名称，以兼容原有代码结构。

> 本仓库仅公开源代码。论文原文件、数据集、预训练权重、训练日志、模型检查点和原始评测结果均未上传。

## 研究动机

在视觉单目标跟踪中，仅依赖初始模板容易受到目标形变、遮挡、视角变化和背景干扰的影响。论文认为，相邻视频帧之间通常具有紧密的时间连续性：上一帧已经确认的目标位置，是当前帧中目标可能出现位置的重要空间先验。

基于这一观察，STPTrack 联合使用两类互补信息：

1. 利用上一帧目标位置提供空间引导，使模型优先关注当前搜索区域中的高概率位置；
2. 利用历史目标的关键外观特征，为当前帧提供跨帧视觉参照，增强模型应对外观变化的能力。

当目标运动平稳时，位置先验能够提高定位效率；当目标快速移动、位置先验出现偏差时，历史外观特征可辅助模型重新识别目标。

## 方法概述

论文中的 STPTrack 由主干网络、时空感知模块、跨帧特征注入模块和目标预测头组成。

### 1. 主干网络

模型使用 Vision Transformer 对以下信息进行联合特征学习：

- 模板图像 `Z`：提供目标的参考外观；
- 搜索区域 `X`：用于定位当前帧中的目标；
- 时空先验 `R`：由相邻帧的目标状态生成；
- 在线传播标记：传递历史帧中的目标判别信息。

模板、搜索区域和时空先验经过标记化后，被拼接为统一序列并送入 ViT 编码器。项目同时采用候选筛除机制，在编码器的第 3、6、9 层逐步抑制低相关性的搜索区域标记，以减少背景干扰。

对应配置位于：

```text
experiments/odtrack/baseline.yaml
```

### 2. 时空感知模块

在处理当前帧 `T` 之前，模型读取上一帧 `T-1` 的预测边界框，并提取其中心坐标：

```math
(x_{T-1}, y_{T-1}) = Center(B_{T-1})
```

随后，以该中心为参考，在当前搜索区域中裁剪固定大小的局部区域，形成空间先验：

```math
R_T = X_T\left[y_{T-1}-\frac{h}{2}:y_{T-1}+\frac{h}{2},
x_{T-1}-\frac{w}{2}:x_{T-1}+\frac{w}{2}\right]
```

裁剪过程包含边界约束，避免目标靠近图像边缘时发生索引越界。生成的先验信息与模板、搜索区域及在线传播标记共同参与后续注意力计算。

相关实现主要位于：

```text
lib/models/odtrack/vit_ce.py
lib/models/layers/head.py
```

### 3. 跨帧特征注入模块

论文使用候选筛除过程产生的注意力关系，从上一帧中选择与目标最相关的搜索区域标记。设第 `L` 层的模板中心特征为 `h_center`，搜索区域特征为 `F`，其相似度为：

```math
s_{T-1}=Similarity\left(h_{center}^{Z,T-1},F_{(L)}^{X,T-1}\right)
```

根据相似度选择得分最高的 `K` 个标记：

```math
I_{topK}=Top(s_{T-1},K)
```

这些标记用于表示上一帧中最关键的目标外观信息，并作为历史视觉参照参与当前帧的模板与搜索区域建模。项目还通过在线 track query 和多参考帧机制传播历史信息。

候选筛除与时序传播相关代码位于：

```text
lib/models/layers/attn_blocks.py
lib/models/odtrack/odtrack.py
lib/models/odtrack/vit_ce.py
```

### 4. 预测头与损失函数

STPTrack 使用中心预测头生成：

- 分类得分图；
- 边界框尺寸图；
- 目标中心偏移图。

训练目标由 Focal Loss、L1 Loss 和 GIoU Loss 组成：

```math
L=L_{cls}+5L_1+2L_{GIoU}
```

相关实现位于：

```text
lib/models/layers/head.py
lib/train/actors/odtrack.py
lib/train/train_script.py
```

## 论文报告结果

论文在 GOT-10K、LaSOT、LaSOT-ext 和 TrackingNet 上进行了评测。

| 数据集 | AUC / AO | 归一化精度 | 精度 / SR0.5 | SR0.75 |
|:---:|:---:|:---:|:---:|:---:|
| TrackingNet | 80.8 AUC | 85.6 | 78.6 | — |
| LaSOT | 69.6 AUC | 80.3 | 75.5 | — |
| LaSOT-ext | 48.7 AUC | 60.0 | 54.6 | — |
| GOT-10K | 74.1 AO | — | 85.9 | 71.7 |

其中，GOT-10K 实验仅使用其官方训练集进行训练，遵循该数据集的评测规范。

论文中的 GOT-10K 消融实验如下：

| 设置 | AO | SR0.5 | SR0.75 |
|:---|:---:|:---:|:---:|
| 基础模型 | 73.0 | 74.9 | 70.3 |
| 基础模型 + 时空感知模块 | 73.7 | 85.9 | 70.6 |
| 基础模型 + 跨帧特征注入模块 | 73.6 | 85.5 | 71.2 |
| 完整 STPTrack | 74.1 | 85.9 | 71.7 |

## 项目结构

```text
STPTRACK/
├── experiments/odtrack/     # 实验配置
├── lib/models/odtrack/      # ViT 主干与跟踪模型
├── lib/models/layers/       # 注意力、候选筛除和预测头
├── lib/train/               # 数据处理、损失函数与训练器
├── lib/test/                # 跟踪器与评测代码
├── tracking/                # 训练、测试、分析和演示入口
├── assets/                  # README 静态资源
└── install.sh               # 环境安装脚本
```

## 环境安装

项目开发环境为 Python 3.8 和 CUDA 11.3：

```bash
conda create -n stptrack python=3.8
conda activate stptrack
bash install.sh
```

## 数据准备

将数据集放置在项目的 `data/` 目录下：

```text
data/
├── lasot/
├── got10k/
│   ├── train/
│   ├── val/
│   └── test/
├── coco/
│   ├── annotations/
│   └── images/
└── trackingnet/
    ├── TRAIN_0/
    ├── ...
    ├── TRAIN_11/
    └── TEST/
```

数据集目录已加入 `.gitignore`，需要从相应数据集的官方网站单独下载。

## 配置本地路径

运行以下命令生成本机训练和测试路径配置：

```bash
python tracking/create_default_local_file.py \
  --workspace_dir . \
  --data_dir ./data \
  --save_dir ./output
```

生成的下列本地文件不会上传到 GitHub：

```text
lib/train/admin/local.py
lib/test/evaluation/local.py
```

## 预训练权重

下载 [MAE ViT-Base 预训练权重](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth)，并放置到：

```text
pretrained_networks/mae_pretrain_vit_base.pth
```

`pretrained_networks/` 已被忽略，权重文件不会被 Git 跟踪。

## 模型训练

使用 4 张 GPU 训练完整配置：

```bash
python tracking/train.py \
  --script odtrack \
  --config baseline \
  --save_dir ./output \
  --mode multiple \
  --nproc_per_node 4 \
  --use_wandb 0
```

仅使用 GOT-10K 训练时：

```bash
python tracking/train.py \
  --script odtrack \
  --config baseline256_got \
  --save_dir ./output \
  --mode multiple \
  --nproc_per_node 4 \
  --use_wandb 0
```

其他实验配置可在 `experiments/odtrack/` 中查看。如果需要使用 Weights & Biases 记录训练过程，将 `--use_wandb` 设置为 `1`。

## 测试与评估

### LaSOT

```bash
python tracking/test.py odtrack baseline \
  --dataset_name lasot \
  --runid 300 \
  --threads 8 \
  --num_gpus 2
```

### LaSOT-ext

```bash
python tracking/test.py odtrack baseline \
  --dataset_name lasot_extension_subset \
  --runid 300 \
  --threads 8 \
  --num_gpus 2
```

运行结果分析前，请先在 `tracking/analysis_results.py` 中设置对应的 `dataset_name` 和 tracker 配置：

```bash
python tracking/analysis_results.py
```

### GOT-10K

```bash
python tracking/test.py odtrack baseline256_got \
  --dataset_name got10k_test \
  --runid 100 \
  --threads 8 \
  --num_gpus 2

python lib/test/utils/transform_got10k.py \
  --tracker_name odtrack \
  --cfg_name baseline256_got_100
```

### TrackingNet

```bash
python tracking/test.py odtrack baseline \
  --dataset_name trackingnet \
  --runid 300 \
  --threads 8 \
  --num_gpus 2

python lib/test/utils/transform_trackingnet.py \
  --tracker_name odtrack \
  --cfg_name baseline_300
```

### 模型复杂度与速度

```bash
python tracking/profile_model.py --script odtrack --config baseline
```

## 论文信息

- 中文题目：时空感知单目标跟踪算法
- 英文题目：STPTrack: A Spatio-Temporal Perception Framework for Visual Object Tracking

论文当前版本中的作者、单位、DOI 等信息尚未完善，因此本 README 不编造 STPTrack 的 BibTeX。相关信息确定后，可在此处补充正式引用。

本项目基于 ODTrack 开发，使用相关代码时请同时引用 ODTrack：

```bibtex
@inproceedings{zheng2024odtrack,
  title={ODTrack: Online Dense Temporal Token Learning for Visual Tracking},
  author={Yaozong Zheng and Bineng Zhong and Qihua Liang and Zhiyi Mo and Shengping Zhang and Xianxian Li},
  booktitle={AAAI},
  year={2024}
}
```

## 致谢

感谢 [ODTrack](https://github.com/GXNU-ZhongLab/ODTrack)、[OSTrack](https://github.com/botaoye/OSTrack)、[STARK](https://github.com/researchmm/Stark) 和 [PyTracking](https://github.com/visionml/pytracking) 的开源工作。

## 开源许可

本项目遵循 [LICENSE](LICENSE) 中的许可协议。
