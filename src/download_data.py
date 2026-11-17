"""Download ISIC 2018 Task 3 (HAM10000) from the official challenge bucket.

Required: training images + ground truth.
Optional: lesion groupings (used for a leakage-free split) and the official
test images + ground truth. If an optional file cannot be downloaded the
pipeline falls back automatically (see splits.py).
"""
import argparse
import os
import urllib.request
import zipfile

BASE = "https://isic-challenge-data.s3.amazonaws.com/2018/"
REQUIRED = ["ISIC2018_Task3_Training_Input.zip", "ISIC2018_Task3_Training_GroundTruth.zip"]
OPTIONAL = ["ISIC2018_Task3_Training_LesionGroupings.csv",
            "ISIC2018_Task3_Test_Input.zip", "ISIC2018_Task3_Test_GroundTruth.zip"]


def fetch(name, out_dir):
    target = os.path.join(out_dir, name)
    marker = target + ".done"
    if os.path.exists(marker):
        print("already have", name)
        return True
    print("downloading", name)
    urllib.request.urlretrieve(BASE + name, target)
    if name.endswith(".zip"):
        with zipfile.ZipFile(target) as zf:
            zf.extractall(out_dir)
        os.remove(target)
    open(marker, "w").close()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    for name in REQUIRED:
        fetch(name, args.data_dir)
    for name in OPTIONAL:
        try:
            fetch(name, args.data_dir)
        except Exception as exc:                      # noqa: BLE001
            print(f"optional file {name} not available ({exc}); continuing without it")


if __name__ == "__main__":
    main()
