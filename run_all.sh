#!/usr/bin/env bash
set -e
python src/download_data.py
python src/splits.py

# 1) capstone-style baseline: ResNet-50, 224px, class-weighted loss, light augmentation
python src/train.py --name baseline_resnet50 --arch resnet50 --img_size 224 --epochs 20 \
    --lr 1e-4 --weight_decay 1e-4 --loss weighted_ce --aug basic --no_tta
python src/evaluate.py --name baseline_resnet50

# 2) improved: ConvNeXt-Tiny, 384px, balanced sampling, strong augmentation, flip TTA
python src/train.py --name convnext_balanced --arch convnext_tiny --img_size 384 --epochs 25 \
    --lr 1e-4 --weight_decay 0.05 --sampler balanced --label_smoothing 0.05 --aug strong
python src/evaluate.py --name convnext_balanced

python src/make_report.py
python src/gradcam.py --name convnext_balanced
