"""Train / validation / test splits for ISIC 2018 Task 3.

HAM10000 contains several images of the same lesion. A random image-level split
puts near-duplicates of one lesion in both train and test and inflates the
score, so the split is grouped by lesion id whenever the groupings file exists.

Data source: the HAM10000 metadata file (same images as ISIC 2018 Task 3, with
lesion ids) is used when present, e.g. as a Kaggle dataset input; otherwise the
ISIC challenge files downloaded by download_data.py.

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


KAGGLE_INPUT = "/kaggle/input"


def _roots(data_dir):
    return [r for r in (data_dir, KAGGLE_INPUT) if os.path.isdir(r)]


def _find(data_dir, pattern):
    for root in _roots(data_dir):
        hits = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
        if hits:
            return hits[0]
    return None


def _image_index(folder_roots):
    """Map image id (file stem) -> path for every .jpg under the given roots."""
    index = {}
    for root in folder_roots:
        for path in glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True):
            index.setdefault(os.path.splitext(os.path.basename(path))[0], path)
    return index


def _read_ham10000(meta_csv):
    """HAM10000 metadata (same 10,015 images as ISIC 2018 Task 3, with lesion ids)."""
    meta = pd.read_csv(meta_csv).drop_duplicates("image_id")
    index = _image_index([os.path.dirname(meta_csv)])
    meta["path"] = meta["image_id"].map(index)
    missing = meta["path"].isna().sum()
    if missing:
        print(f"warning: {missing} images listed in the metadata were not found and are skipped")
    meta = meta.dropna(subset=["path"])
    meta["label"] = meta["dx"].str.upper().map({c: i for i, c in enumerate(CLASSES)})
    meta = meta.dropna(subset=["label"])
    meta["label"] = meta["label"].astype(int)
    return meta.rename(columns={"image_id": "image"})[["image", "path", "label", "lesion_id"]]


def _read_ground_truth(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    df["label"] = df[CLASSES].values.argmax(axis=1)
    df["path"] = df["image"].map(lambda i: os.path.join(image_dir, f"{i}.jpg"))
    return df[["image", "path", "label"]]


def build_splits(data_dir="data", seed=42):
    ham_csv = _find(data_dir, "HAM10000_metadata.csv")
    train_csv = _find(data_dir, "ISIC2018_Task3_Training_GroundTruth.csv")
    if ham_csv:
        df = _read_ham10000(ham_csv)
        grouping = "lesion"
    elif train_csv:
        df = _read_ground_truth(train_csv, os.path.join(os.path.dirname(os.path.dirname(train_csv)),
                                                        "ISIC2018_Task3_Training_Input"))
        grouping = None
    else:
        raise FileNotFoundError("No dataset found. Add the HAM10000 dataset as notebook input "
                                "or run download_data.py.")

    groups_csv = None if grouping else _find(data_dir, "ISIC2018_Task3_Training_LesionGroupings.csv")
    if grouping:
        pass
    elif groups_csv:
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
    test_dir = os.path.join(os.path.dirname(os.path.dirname(test_csv)),
                            "ISIC2018_Task3_Test_Input") if test_csv else None
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
