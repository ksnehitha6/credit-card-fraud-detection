# Utility functions can be added here for future use.
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

OUTPUT_DIR = "outputs"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _maybe_roc_auc(y_true, y_proba):
    if y_proba is None:
        return None
    try:
        return float(roc_auc_score(y_true, y_proba))
    except Exception:
        return None

def save_metrics(y_true, y_pred, model_name: str, y_proba=None):
    """Compute metrics, print to console, and save JSON report."""
    ensure_dir(OUTPUT_DIR)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _maybe_roc_auc(y_true, y_proba),
        "report": classification_report(y_true, y_pred, zero_division=0)
    }
    print(f"\n=== {model_name} ===")
    for k, v in metrics.items():
        if k != "report":
            print(f"{k:>8}: {v}")
    print("\n" + metrics["report"])

    out_path = os.path.join(OUTPUT_DIR, f"metrics_{model_name}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[saved] {out_path}")

def plot_confusion_matrix(y_true, y_pred, model_name: str):
    """Plot and save confusion matrix as PNG."""
    ensure_dir(OUTPUT_DIR)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
        xticklabels=[0, 1], yticklabels=[0, 1],
        xlabel="Predicted label", ylabel="True label",
        title=f"Confusion Matrix — {model_name}"
    )
    # Put numbers on the cells
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    out_img = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name}.png")
    fig.savefig(out_img, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_img}")
