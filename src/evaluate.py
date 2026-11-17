"""Evaluate saved probabilities: metrics, bootstrap confidence intervals and a
melanoma operating point whose threshold is chosen on the validation set."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)

from splits import CLASSES, MALIGNANT

MEL = CLASSES.index("MEL")


def onehot(y):
    return np.eye(len(CLASSES))[y]


def summary_metrics(y, p):
    pred = p.argmax(1)
    out = {"accuracy": accuracy_score(y, pred),
           "balanced_accuracy": balanced_accuracy_score(y, pred),
           "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
           "melanoma_sensitivity": float((pred[y == MEL] == MEL).mean())}
    try:
        out["macro_auc"] = roc_auc_score(onehot(y), p, average="macro")
    except ValueError:                      # a class is missing from this sample
        out["macro_auc"] = float("nan")
    return {k: float(v) for k, v in out.items()}


def bootstrap(y, p, fn, n=1000, seed=42):
    """95% percentile intervals from n resamples of the test set."""
    rng = np.random.default_rng(seed)
    rows = [fn(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return {k: [float(np.nanpercentile([r[k] for r in rows], 2.5)),
                float(np.nanpercentile([r[k] for r in rows], 97.5))] for k in rows[0]}


def per_class(y, p):
    pred, cm, out = p.argmax(1), confusion_matrix(y, p.argmax(1), labels=range(len(CLASSES))), {}
    for i, name in enumerate(CLASSES):
        tp, fn = cm[i, i], cm[i].sum() - cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - tp - fn - fp
        out[name] = {"support": int(cm[i].sum()),
                     "sensitivity": float(tp / max(1, tp + fn)),
                     "specificity": float(tn / max(1, tn + fp)),
                     "precision": float(tp / max(1, tp + fp)),
                     "f1": float(f1_score(y == i, pred == i, zero_division=0)),
                     "auc": float(roc_auc_score(y == i, p[:, i])) if 0 < (y == i).sum() < len(y) else None,
                     "average_precision": float(average_precision_score(y == i, p[:, i]))}
    return out, cm


def threshold_for_sensitivity(y_val, score_val, target):
    """Highest threshold whose validation sensitivity is at least `target`."""
    pos = np.sort(score_val[y_val])[::-1]
    k = int(np.ceil(target * len(pos)))
    return float(pos[max(0, k - 1)])


def binary_point(y_pos, score, thr):
    flag = score >= thr
    return {"sensitivity": float(flag[y_pos].mean()), "specificity": float((~flag[~y_pos]).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--target_sensitivity", type=float, default=0.90)
    args = ap.parse_args()

    d = np.load(os.path.join(args.out_dir, f"{args.name}_probs.npz"), allow_pickle=False)
    yv, pv, yt, pt = d["val_labels"], d["val_probs"], d["test_labels"], d["test_probs"]
    with open(os.path.join(args.out_dir, f"{args.name}_train.json")) as fh:
        train_info = json.load(fh)

    classes, cm = per_class(yt, pt)
    result = {"name": args.name, "protocol": train_info["protocol"], "n_test": int(len(yt)),
              "config": train_info["config"], "train_minutes": train_info["train_minutes"],
              "best_val_balanced_acc": train_info["best_val_balanced_acc"],
              "test": summary_metrics(yt, pt), "test_ci95": bootstrap(yt, pt, summary_metrics),
              "per_class": classes, "confusion_matrix": cm.tolist(),
              "mean_average_precision": float(np.mean([c["average_precision"] for c in classes.values()]))}

    # Melanoma screening view: flag an image when P(MEL) >= t, with t fixed on validation data.
    thr = threshold_for_sensitivity(yv == MEL, pv[:, MEL], args.target_sensitivity)
    point = lambda y, p: binary_point(y == MEL, p[:, MEL], thr)          # noqa: E731
    result["melanoma_operating_point"] = {
        "target_validation_sensitivity": args.target_sensitivity, "threshold": thr,
        "validation": point(yv, pv), "test": point(yt, pt), "test_ci95": bootstrap(yt, pt, point),
        "test_auc": float(roc_auc_score(yt == MEL, pt[:, MEL]))}

    # Malignant (MEL, BCC, AKIEC) vs benign, using summed malignant probability.
    mal = [CLASSES.index(c) for c in MALIGNANT]
    ym_v, sm_v = np.isin(yv, mal), pv[:, mal].sum(1)
    ym_t, sm_t = np.isin(yt, mal), pt[:, mal].sum(1)
    thr_m = threshold_for_sensitivity(ym_v, sm_v, args.target_sensitivity)
    result["malignant_vs_benign"] = {"threshold": thr_m, "test_auc": float(roc_auc_score(ym_t, sm_t)),
                                     "test": binary_point(ym_t, sm_t, thr_m)}

    with open(os.path.join(args.out_dir, f"{args.name}.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{args.name}: row-normalised confusion matrix (test)", fontsize=10)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"{args.name}_confusion.png"), dpi=160)
    plt.close(fig)

    t, ci = result["test"], result["test_ci95"]
    print(f"[{args.name}] acc {t['accuracy']:.4f}  balanced acc {t['balanced_accuracy']:.4f} "
          f"({ci['balanced_accuracy'][0]:.3f}-{ci['balanced_accuracy'][1]:.3f})  "
          f"macro AUC {t['macro_auc']:.4f}  MEL sens {t['melanoma_sensitivity']:.4f}")
    mp = result["melanoma_operating_point"]["test"]
    print(f"   melanoma at val-tuned threshold: sens {mp['sensitivity']:.4f} spec {mp['specificity']:.4f}")


if __name__ == "__main__":
    main()
