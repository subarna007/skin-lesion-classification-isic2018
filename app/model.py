"""Loads the trained classifier and produces probabilities and Grad-CAM maps.

Preprocessing is imported from src/data.py so it is identical to training.
"""
import os
import sys

import numpy as np
import timm
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from data import build_transforms  # noqa: E402


def _target_layer(model):
    if hasattr(model, "stages"):          # ConvNeXt
        return model.stages[-1].blocks[-1]
    if hasattr(model, "layer4"):          # ResNet
        return model.layer4[-1]
    raise ValueError("unknown architecture for Grad-CAM")


class Classifier:
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.arch, self.img_size, self.classes = ckpt["arch"], ckpt["img_size"], ckpt["classes"]
        self.model = timm.create_model(self.arch, pretrained=False, num_classes=len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])        # strict on purpose
        self.model.to(self.device).eval()
        self.transform = build_transforms(self.img_size, train=False)

    def _tensor(self, pil_image):
        return self.transform(pil_image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, pil_image):
        """Class probabilities, averaged over the four flips used at evaluation."""
        x = self._tensor(pil_image)
        views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)]
        probs = sum(torch.softmax(self.model(v), dim=1) for v in views) / len(views)
        return probs[0].cpu().numpy()

    def gradcam(self, pil_image, class_index):
        """Heatmap overlay (uint8 RGB) for the given class."""
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        x = self._tensor(pil_image)
        with GradCAM(model=self.model, target_layers=[_target_layer(self.model)]) as cam:
            heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(int(class_index))])[0]
        rgb = np.asarray(pil_image.resize((self.img_size, self.img_size)), dtype=np.float32) / 255.0
        return show_cam_on_image(rgb, heat, use_rgb=True)
