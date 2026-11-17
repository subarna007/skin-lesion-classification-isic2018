"""Class-specific Grad-CAM for a few test images per class.

The heatmap is computed for the class the model predicted, so it explains the
prediction that is actually shown (requires `pip install grad-cam`).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from data import build_transforms
from splits import CLASSES


def target_layer(model):
    if hasattr(model, "stages"):          # ConvNeXt
        return model.stages[-1].blocks[-1]
    if hasattr(model, "layer4"):          # ResNet
        return model.layer4[-1]
    raise ValueError("add the last convolutional block for this architecture")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--per_class", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(args.ckpt_dir, f"{args.name}.pth"), map_location=device)
    model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])          # strict: fails loudly on any mismatch
    model.to(device).eval()
    size, tf = ckpt["img_size"], build_transforms(ckpt["img_size"], False)

    df = pd.read_csv(os.path.join(args.out_dir, "splits.csv"))
    test = df[df["split"] == "test"]
    picks = test.groupby("label").head(args.per_class)

    cam = GradCAM(model=model, target_layers=[target_layer(model)])
    fig, axes = plt.subplots(args.per_class * 2, len(CLASSES), figsize=(2.1 * len(CLASSES), 4.4 * args.per_class))
    for col, cls in enumerate(range(len(CLASSES))):
        rows = picks[picks["label"] == cls]
        for k, (_, row) in enumerate(rows.iterrows()):
            img = Image.open(row["path"]).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = int(model(x).argmax(1))
            heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
            rgb = np.asarray(img.resize((size, size)), dtype=np.float32) / 255.0
            axes[2 * k, col].imshow(rgb)
            axes[2 * k, col].set_title(f"true {CLASSES[cls]}", fontsize=8)
            axes[2 * k + 1, col].imshow(show_cam_on_image(rgb, heat, use_rgb=True))
            axes[2 * k + 1, col].set_title(f"pred {CLASSES[pred]}", fontsize=8,
                                           color="black" if pred == cls else "#d03b3b")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"{args.name}_gradcam.jpg"), dpi=150, pil_kwargs={"quality": 85})
    print("saved Grad-CAM grid")


if __name__ == "__main__":
    main()
