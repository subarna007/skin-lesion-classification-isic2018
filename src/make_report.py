"""Combine results/*.json into a table (results/results.md, README) and a
per-class sensitivity figure."""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from splits import CLASSES

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
COLORS = ["#9aa7b1", "#0e6b66", "#2a78d6", "#eda100"]
NICE = {"baseline_resnet50": "ResNet-50 baseline", "convnext_balanced": "ConvNeXt-Tiny"}


def ci(r, key):
    lo, hi = r["test_ci95"][key]
    return f"{r['test'][key]:.3f} ({lo:.3f}-{hi:.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--readme", default="README.md")
    args = ap.parse_args()

    recs = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        if path.endswith("_train.json"):
            continue
        recs.append(json.load(open(path)))
    if not recs:
        raise SystemExit("no results found")
    recs.sort(key=lambda r: r["test"]["balanced_accuracy"])

    lines = ["| Model | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUC | Melanoma sensitivity | DF sensitivity |",
             "|---|---|---|---|---|---|---|"]
    for r in recs:
        c = r["config"]
        lines.append(f"| {NICE.get(r['name'], r['name'])} ({c['img_size']} px) | {ci(r, 'accuracy')} | "
                     f"{ci(r, 'balanced_accuracy')} | {r['test']['macro_f1']:.3f} | {ci(r, 'macro_auc')} | "
                     f"{ci(r, 'melanoma_sensitivity')} | {r['per_class']['DF']['sensitivity']:.3f} |")
    lines += ["", f"Test set: {recs[0]['n_test']:,} images, a held-out split grouped by lesion. "
                  "Brackets are 95% bootstrap intervals (1,000 resamples).", "",
              "Melanoma screening view (flag when P(MEL) is above a threshold fixed on the validation set):", "",
              "| Model | Threshold | Test sensitivity | Test specificity | Melanoma AUC |", "|---|---|---|---|---|"]
    for r in recs:
        m = r["melanoma_operating_point"]
        lines.append(f"| {NICE.get(r['name'], r['name'])} | {m['threshold']:.3f} | {m['test']['sensitivity']:.3f} "
                     f"({m['test_ci95']['sensitivity'][0]:.3f}-{m['test_ci95']['sensitivity'][1]:.3f}) | "
                     f"{m['test']['specificity']:.3f} | {m['test_auc']:.3f} |")
    table = "\n".join(lines)

    with open(os.path.join(args.results_dir, "results.md"), "w") as fh:
        fh.write(table + "\n")
    if os.path.exists(args.readme):
        text = open(args.readme, encoding="utf-8").read()
        if START in text and END in text:
            text = re.sub(re.escape(START) + r".*?" + re.escape(END),
                          lambda _: f"{START}\n{table}\n{END}", text, flags=re.S)
            open(args.readme, "w", encoding="utf-8").write(text)

    fig, ax = plt.subplots(figsize=(8.5, 3.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    x, width = np.arange(len(CLASSES)), 0.8 / len(recs)
    for i, r in enumerate(recs):
        vals = [r["per_class"][c]["sensitivity"] for c in CLASSES]
        ax.bar(x + (i - (len(recs) - 1) / 2) * width, vals, width * 0.92, label=NICE.get(r["name"], r["name"]),
               color=COLORS[i % len(COLORS)])
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Sensitivity on the test set", color="#52514e")
    ax.yaxis.grid(True, color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Per-class sensitivity on the test set", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, ncol=len(recs), loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "fig_per_class_sensitivity.png"), dpi=200)
    plt.close(fig)
    print(table)


if __name__ == "__main__":
    main()
