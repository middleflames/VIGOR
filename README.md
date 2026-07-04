# VIGOR: Visual Grounding via Heterogeneous Representation Learning and Hierarchical Reasoning of Human-to-Vehicle Commands

Official materials for the ICRA 2026 paper:

**Visual Grounding via Heterogeneous Representation Learning and Hierarchical Reasoning of Human-to-Vehicle Commands (VIGOR)**

## Overview

VIGOR studies visual grounding for human-to-vehicle commands. This repository provides the reference implementation (model, training and evaluation code) alongside the paper presentation materials and citation information.

## Repository Structure

```
run.py                   # single-node launcher (wraps torch.distributed.launch)
train.py                 # train / evaluate entry point (XVLM_POSLIDAR)
configs/
  vigor_talk2car.yaml    # experiment config; all data paths rooted at ./data
  config_swinB_384.json  # Swin-B vision-encoder config
  config_bert.json       # BERT text-encoder config
models/                  # VIGOR model (XVLM_POSLIDAR) + X-VLM backbone
dataset/                 # talk2car + LiDAR grounding dataset and evaluation
utils/, refTools/        # metrics, distributed helpers, REFER API
data/                    # datasets go here (see data/README.md)
```

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Tested with Python 3.9 and CUDA 11.8. Place the datasets under `data/` following [`data/README.md`](data/README.md).

## Usage

The simplest entry point is the `run.py` launcher (single node, one or more GPUs):

```bash
# Evaluate a trained checkpoint on GPU 0
python run.py --checkpoint /path/to/checkpoint_best.pth \
    --output_dir output/vigor --gpus 0 --evaluate

# Train on GPUs 0,1,2, fine-tuning from the X-VLM bbox pre-trained checkpoint
python run.py --checkpoint /path/to/xvlm_pretrained.pth \
    --output_dir output/vigor --gpus 0,1,2 --load_bbox_pretrain
```

`run.py` forwards to `train.py` through `torch.distributed.launch`; call `train.py`
directly for full control:

```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py \
    --config configs/vigor_talk2car.yaml \
    --checkpoint /path/to/checkpoint_best.pth \
    --output_dir output/vigor --evaluate
```

## Figures
### Figure 1: System Overview

![System overview of the VIGOR framework](figures/system_overview_vigor.png)


### Figure 2: POS-Tag Data Visualization

![POS-tag based data visualization for human-to-vehicle commands](figures/data_show_pos_tag.png)


## TODO List
- [x] Best Checkpoint and necessary meta data (2026-Jul-4)
- [x] Model Update (2026-Jul-4)
- [x] Initial Repo


## Abstract

With the proliferation of autonomous vehicles (AVs) and their increasing interaction and communication with riders, grounding or locating the visual objects of interest (OoIs), such as concerned pedestrians and other traffic participants, based on human riders' natural language communication, e.g., vocal commands, is essential for increasing the efficiency, effectiveness, reliability, and safety of AVs in following riders' reasonable commands and preferences.

There are several technical challenges in achieving visual grounding for such human-to-vehicle commanding (HVC) scenes, including: (1) how to fuse heterogeneous sensor modalities, i.e., visual object information, textual contexts, and situation awareness, such as information obtained from light detection and ranging; (2) how to discern opaque commands in human natural language; and (3) how to reason about the relative positions of the OoIs within the visual modality.

To meet these challenges, we propose VIGOR, a visual grounding approach based on heterogeneous modality learning and hierarchical reasoning for HVC scenes. First, we design a heterogeneous modality learning approach to incorporate visual, textual, and situational modalities, and learn their cross-modality representations to identify important information for visual grounding. Then, VIGOR performs hierarchical reasoning at the object and context levels, and differentiates the OoIs in complex traffic environments that relate to natural language commands.

Finally, we conduct extensive experimental studies on a total of 12,037 HVC scenes, demonstrating that VIGOR achieves higher accuracy than state-of-the-art approaches, often by more than 14.81%, in terms of Intersection over Union (IoU) in grounding the OoIs in complex HVC scenes, including low-visibility scenes.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{vigor2026visual,
  title     = {Visual Grounding via Heterogeneous Representation Learning and Hierarchical Reasoning of Human-to-Vehicle Commands},
  author    = {Wang, H. and He, S. and Shin, K. G.},
  booktitle = {Proceedings of 2026 IEEE International Conference on Robotics and Automation (ICRA)},
  address   = {Vienna, Austria},
  month     = jun,
  year      = {2026},
  note      = {June 1-5, 2026}
}
```