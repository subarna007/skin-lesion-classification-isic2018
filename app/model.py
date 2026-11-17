"""Loads the trained classifier: probabilities, embeddings and Grad-CAM maps.

Preprocessing is imported from src/data.py so it matches training exactly.
"""
import os
import sys

import cv2
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
        """Class probabilities averaged over the four flips used at evaluation,
        plus the L2-normalised image embedding (pre-logits features)."""
        x = self._tensor(pil_image)
        views = [x, x.flip(3), x.flip(2), x.flip(2).flip(3)]
        probs = sum(torch.softmax(self.model(v), dim=1) for v in views) / len(views)
        emb = self.model.forward_head(self.model.forward_features(x), pre_logits=True)
        emb = torch.nn.functional.normalize(emb.float(), dim=1)
        return probs[0].cpu().numpy(), emb[0].cpu().numpy()

    def heatmaps(self, pil_image, class_indices):
        """Colour Grad-CAM heatmap (uint8 RGB, img_size x img_size) for each class."""
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        x = self._tensor(pil_image)
        out = {}
        with GradCAM(model=self.model, target_layers=[_target_layer(self.model)]) as cam:
            for idx in class_indices:
                heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(int(idx))])[0]
                colour = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
                out[int(idx)] = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
        return out
