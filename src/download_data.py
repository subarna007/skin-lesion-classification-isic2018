"""Get the ISIC 2018 Task 3 / HAM10000 images.

If the HAM10000 dataset is already available (for example attached as a Kaggle
dataset under /kaggle/input), nothing is downloaded. Otherwise the files are
fetched from the ISIC challenge bucket and checked before they are unpacked.
"""
import argparse
import glob
import os
import sys
import urllib.request
import zipfile

BASE = "https://isic-challenge-data.s3.amazonaws.com/2018/"
REQUIRED = ["ISIC2018_Task3_Training_Input.zip", "ISIC2018_Task3_Training_GroundTruth.zip"]
OPTIONAL = ["ISIC2018_Task3_Training_LesionGroupings.csv",
            "ISIC2018_Task3_Test_Input.zip", "ISIC2018_Task3_Test_GroundTruth.zip"]


def already_available(data_dir):
    for root in (data_dir, "/kaggle/input"):
        for name in ("HAM10000_metadata.csv", "ISIC2018_Task3_Training_GroundTruth.csv"):
            hits = glob.glob(os.path.join(root, "**", name), recursive=True)
            if hits:
                return hits[0]
    return None


def fetch(name, out_dir):
    target = os.path.join(out_dir, name)
    if os.path.exists(target + ".done"):
        return
    print("downloading", name)
    urllib.request.urlretrieve(BASE + name, target)
    if name.endswith(".zip"):
        if not zipfile.is_zipfile(target):
            with open(target, "rb") as fh:
                head = fh.read(300)
            os.remove(target)
            raise RuntimeError(f"{name}: server did not return a zip file. Start of response: {head!r}")
        with zipfile.ZipFile(target) as zf:
            zf.extractall(out_dir)
        os.remove(target)
    open(target + ".done", "w").close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()

    found = already_available(args.data_dir)
    if found:
        print("dataset already available:", found)
        return

    os.makedirs(args.data_dir, exist_ok=True)
    try:
        for name in REQUIRED:
            fetch(name, args.data_dir)
    except Exception as exc:                                  # noqa: BLE001
        sys.exit(f"Download failed: {exc}\n"
                 "Alternative: attach the HAM10000 dataset as an input "
                 "(Kaggle: 'Skin Cancer MNIST: HAM10000') and run again.")
    for name in OPTIONAL:
        try:
            fetch(name, args.data_dir)
        except Exception as exc:                              # noqa: BLE001
            print(f"optional file {name} not available ({exc}); continuing without it")


if __name__ == "__main__":
    main()
