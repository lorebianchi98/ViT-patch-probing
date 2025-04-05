# ProbeIt! — Linear Probing of ViT Patch Representations

This repository explores the **semantic richness of patch-level representations** from Vision Transformer (ViT) models—including **CLIP, DINO, MAE, and DINOv2**—through **linear probing** on a semantic segmentation dataset, specifically [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/).

We aim to answer the question: _"How semantically meaningful are the patch embeddings of pretrained ViTs?"_

---

##  Method Overview

Given a ViT model and the Pascal VOC segmentation dataset:

1. **Patch Labeling**: Each patch is assigned a class via **majority voting** over its pixels (each pixel has a class label).
2. **Linear Probing**: A linear classifier is trained to predict patch-level classes using frozen patch embeddings from the ViT model.
3. **Evaluation**: We assess the learned layer’s performance on the Pascal VOC validation set.

The linear layer has shape **(EMBED_DIM, NUM_CLASSES)** and is trained using frozen features from the ViT backbone.

---

## Installation

```bash
conda create --name probe_it python=3.9 -y
conda activate probe_it

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt
```

---

## Dataset Setup

This repo uses the **Pascal VOC 2012** dataset.

- 📥 Download it from [this link](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).
- 📦 Follow the [MMSegmentation preparation guide](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-voc) to set it up properly.

---

## Running the Probing Pipeline

To run a linear probing experiment:

```bash
python main.py --config path/to/config.yaml --out_dir results/
```

Make sure to define your backbone and training parameters in the YAML config.

---

## Results

Below are accuracy results of linear probing on various ViT backbones.  
Training used: **batch size = 16**, **Adam optimizer**, **learning rate = 5e-3**, **3 epochs**, and **32×32 patch resolution**.

| **Visual Backbone**              | **ViT-S** | **ViT-B** | **ViT-L** |
|----------------------------------|:--------:|:--------:|:--------:|
| MAE                              |    -     |   0.61   |   0.62   |
| CLIP                             |    -     |   0.91   |   0.90   |
| DINO                             |   0.65   |   0.77   |    -     |
| DINOv2 (without registers)       |   0.95   |   0.96   |   0.95   |
| **DINOv2 (with registers)** 🔥   | **0.97** | **0.97** | **0.96** |

---

## 📌 TODO / Coming Soon

- [ ] Add support for other segmentation datasets (e.g., ADE20K, Cityscapes)

---

## Contributions

Feel free to open issues or submit pull requests for improvements, new backbones, or dataset integrations.

---
##  Support

If this repo helped you, or you used it in your work, **drop a star** ⭐!