"""Train / validation / test splits for ISIC 2018 Task 3.

HAM10000 contains several images of the same lesion. A random image-level split
puts near-duplicates of one lesion in both train and test and inflates the
score, so the split is grouped by lesion id whenever the groupings file exists.

Protocol A (preferred): official 1,512-image test set as the test set;
    the 10,015 training images are split ~85/15 into train/val by lesion.
Protocol B (fallback, when the official test labels are not available):
    the 10,015 images are split ~70/15/15 into train/val/test by lesion.
"""
import glob
import os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
MALIGNANT = ["MEL", "BCC", "AKIEC"]          # AKIEC is treated as (pre-)malignant


def _find(data_dir, pattern):
    hits = sorted(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True))
    return hits[0] if hits else None


def _read_ground_truth(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    df["label"] = df[CLASSES].values.argmax(axis=1)
    df["path"] = df["image"].map(lambda i: os.path.join(image_dir, f"{i}.jpg"))
    return df[["image", "path", "label"]]


def build_splits(data_dir="data", seed=42):
    train_csv = _find(data_dir, "ISIC2018_Task3_Training_GroundTruth.csv")
    if train_csv is None:
        raise FileNotFoundError("Training ground truth not found; run download_data.py first")
    train_dir = os.path.join(data_dir, "ISIC2018_Task3_Training_Input")
    df = _read_ground_truth(train_csv, train_dir)

    groups_csv = _find(data_dir, "ISIC2018_Task3_Training_LesionGroupings.csv")
    if groups_csv:
        g = pd.read_csv(groups_csv)
        lesion_col = next(c for c in g.columns if "lesion" in c.lower())
        df = df.merge(g[["image", lesion_col]].rename(columns={lesion_col: "lesion_id"}),
                      on="image", how="left")
        df["lesion_id"] = df["lesion_id"].fillna(df["image"])
        grouping = "lesion"
    else:
        df["lesion_id"] = df["image"]
        grouping = "image"

    test_csv = _find(data_dir, "ISIC2018_Task3_Test_GroundTruth.csv")
    test_dir = os.path.join(data_dir, "ISIC2018_Task3_Test_Input")
    official_test = bool(test_csv) and os.path.isdir(test_dir)

    n_folds = 7 if official_test else 20
    folds = list(StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                 .split(df, df["label"], groups=df["lesion_id"]))
    df["split"] = "train"
    if official_test:
        df.loc[folds[0][1], "split"] = "val"                      # ~14%
        test = _read_ground_truth(test_csv, test_dir)
        test["lesion_id"] = test["image"]
        test["split"] = "test"
        df = pd.concat([df, test], ignore_index=True)
        protocol = "A: official ISIC 2018 test set"
    else:
        for k in (0, 1, 2):
            df.loc[folds[k][1], "split"] = "test"                 # ~15%
        for k in (3, 4, 5):
            df.loc[folds[k][1], "split"] = "val"                  # ~15%
        protocol = "B: internal held-out test split"
    df.attrs["protocol"] = f"{protocol}; grouped by {grouping}"
    return df


def load_or_build_splits(data_dir="data", out_csv="results/splits.csv", seed=42):
    """Build the split once and reuse it, so every model sees identical data."""
    meta = out_csv + ".protocol.txt"
    if os.path.exists(out_csv) and os.path.exists(meta):
        df = pd.read_csv(out_csv)
        df.attrs["protocol"] = open(meta).read().strip()
        return df
    df = build_splits(data_dir, seed)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    with open(meta, "w") as fh:
        fh.write(df.attrs["protocol"])
    return df


if __name__ == "__main__":
    d = load_or_build_splits()
    print(d.attrs["protocol"])
    print(pd.crosstab(d["split"], d["label"].map(lambda i: CLASSES[i])))
