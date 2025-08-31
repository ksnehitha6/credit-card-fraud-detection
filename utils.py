import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score
)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def summarize_metrics(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    roc = None
    if y_proba is not None:
        try:
            roc = roc_auc_score(y_true, y_proba)
        except Exception:
            roc = None
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}

def print_and_save_report(name, y_true, y_pred, out_dir, y_proba=None):
    ensure_dir(out_dir)
    metrics = summarize_metrics(y_true, y_pred, y_proba)
    cm = confusion_matrix(y_true, y_pred)

    # console
    print(f"\n=== {name} ===")
    print("Metrics:", json.dumps({k: (None if v is None else float(v)) for k, v in metrics.items()}, indent=2))
    print("\nClassification Report\n", classification_report(y_true, y_pred, digits=4, zero_division=0))

    # save confusion matrix plot
    fig_path = os.path.join(out_dir, f"confusion_matrix_{name}.png")
    plot_confusion_matrix(cm, title=f"Confusion Matrix - {name}", save_path=fig_path)

    # save json metrics
    with open(os.path.join(out_dir, f"metrics_{name}.json"), "w") as f:
        json.dump({k: (None if v is None else float(v)) for k, v in metrics.items()}, f, indent=2)

def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=[0, 1],
        yticklabels=[0, 1],
        ylabel="True label",
        xlabel="Predicted label",
        title=title
    )

    # text
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
