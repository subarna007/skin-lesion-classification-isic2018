"""Build the similar-case gallery used by the DermAssist web demo.

For a class-balanced sample of *training* images (never validation or test),
stores the model's L2-normalised feature vector and a small thumbnail. The web
app compares an uploaded image with these vectors to show its nearest
neighbours. Output: app/gallery/index.npz and app/gallery/thumbs/*.jpg.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from tqdm.auto import tqdm

from data import build_transforms
from splits import CLASSES


def find_checkpoint(name, ckpt_dir):
    """Look in the checkpoint folder, then in the usual Kaggle locations."""
    filename = f"{name}.pth"
    candidates = [os.path.join(ckpt_dir, filename), os.path.join("/kaggle/working", filename)]
    candidates += sorted(glob.glob(os.path.join("/kaggle/input", "**", filename), recursive=True))
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise SystemExit(f"Could not find {filename}. Looked in {ckpt_dir}/, /kaggle/working/ and /kaggle/input/. "
                     "Pass its location with --ckpt, or train the model again with src/train.py.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="convnext_balanced")
    ap.add_argument("--per_class", type=int, default=120)
    ap.add_argument("--thumb", type=int, default=160)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--ckpt", default=None, help="explicit path to the .pth file")
    ap.add_argument("--out_dir", default="app/gallery")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt or find_checkpoint(args.name, args.ckpt_dir)
    print("using checkpoint", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    tf = build_transforms(ckpt["img_size"], train=False)

    df = pd.read_csv(os.path.join(args.results_dir, "splits.csv"))
    train = df[df["split"] == "train"].drop_duplicates("lesion_id")       # one image per lesion
    sample = (train.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), args.per_class), random_state=args.seed)))

    os.makedirs(os.path.join(args.out_dir, "thumbs"), exist_ok=True)
    embs, labels, images = [], [], []
    with torch.no_grad():
        for _, row in tqdm(sample.iterrows(), total=len(sample), desc="gallery"):
            img = Image.open(row["path"]).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)
            e = model.forward_head(model.forward_features(x), pre_logits=True)
            embs.append(torch.nn.functional.normalize(e.float(), dim=1)[0].cpu().numpy())
            labels.append(int(row["label"]))
            images.append(row["image"])
            w, h = img.size
            s = min(w, h)
            img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)) \
               .resize((args.thumb, args.thumb), Image.LANCZOS) \
               .save(os.path.join(args.out_dir, "thumbs", f"{row['image']}.jpg"), quality=82)

    np.savez(os.path.join(args.out_dir, "index.npz"), embeddings=np.stack(embs).astype(np.float16),
             labels=np.array(labels, dtype=np.int16), images=np.array(images))
    with open(os.path.join(args.out_dir, "README.md"), "w") as fh:
        fh.write("Thumbnails of HAM10000 training images (Tschandl et al., 2018), licensed CC BY-NC 4.0, "
                 "used to show similar cases in the DermAssist demo.\n")
    print(f"gallery: {len(images)} images -> {args.out_dir}")


if __name__ == "__main__":
    main()
