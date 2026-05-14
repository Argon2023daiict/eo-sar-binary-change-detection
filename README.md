[Download best_model.pth](https://drive.google.com/file/d/19bbV7qW_xfbkQNv88nZLK0tJ3r-BMINJ/view?usp=drive_link)

# EO-SAR Binary Change Detection — ChangeFormer

Binary pixel-level change detection on paired Electro-Optical (EO) and Synthetic
Aperture Radar (SAR) satellite imagery using a transformer-based ChangeFormer
architecture with Swin Transformer Tiny backbone.

**Author:** Rahul Kumar


---

## Approach

- ChangeFormer with shared Swin-T encoder for EO and SAR modalities
- Fusion blocks at each encoder scale (concat + ConvBNReLU)
- UNet-style decoder with multi-scale skip connections
- Hybrid BCE + Dice loss for class imbalance
- Weighted random sampler for balanced training
- Test-Time Augmentation (horizontal + vertical flip) at inference
- Strong augmentation pipeline via Albumentations

---

## Results

| Split      | F1     | IoU    | Precision | Recall |
|------------|--------|--------|-----------|--------|
| Validation | 0.8467 | 0.7342 | —         | —      |
| Test       | 0.5818 | 0.4102 | 0.5333    | 0.6399 |

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (trained on NVIDIA Tesla P100 16GB)
- torch==2.2.0
torchvision==0.17.0
timm==0.9.12
albumentations==1.4.0
opencv-python==4.9.0.80
rasterio==1.3.9
numpy==1.26.4
matplotlib==3.8.3
PyYAML==6.0.1
scipy==1.12.0
tqdm==4.66.2
Pillow==10.2.0
scikit-learn==1.4.1

---

## Environment Setup

```bash
conda create -n galaxeye python=3.10 -y
conda activate galaxeye
pip install -r requirements.txt
```

---

## Dataset Structure

```
data/
├── train/
│   ├── pre-event/   ← EO RGB .tif files
│   ├── post-event/  ← SAR single-channel .tif files
│   └── target/      ← binary mask .tif files
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/
```

Update `config.yaml` with your data paths:
```yaml
data:
  train_dir: "data/train"
  val_dir:   "data/val"
  test_dir:  "data/test"
```

---

## Training

```bash
python train.py --config config.yaml
```

---

## Evaluation

```bash
python eval.py \
  --config config.yaml \
  --weights checkpoints_changeformer/best_model.pth \
  --threshold 0.8 \
  --save_dir eval_results
```

---



## Citation / References

1. Bai et al. — *Deep learning for change detection in remote sensing: a review* (2023)
2. Lei et al. — *Remote sensing image change detection using deep learning: a comprehensive survey* (2023)
3. TransUNet++SAR — *Change Detection with Deep Learning about Architectural Ensemble in SAR Images*
4. *Change Detection in Heterogeneous Optical and SAR Images via Deep Homogeneous Feature Fusion*
5. Liu et al. — *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows* (ICCV 2021)
6. Bandara & Patel — *A Transformer-Based Siamese Network for Change Detection* (IGARSS 2022)
7. He et al. — *Deep Residual Learning for Image Recognition* (CVPR 2016)
8. Lee, J.S. — *Digital image enhancement and noise filtering by use of local statistics* (1980)
9. Buslaev et al. — *Albumentations: Fast and Flexible Image Augmentations* (2020)
