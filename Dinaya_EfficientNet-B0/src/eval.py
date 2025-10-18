# src/eval.py
import os, json, argparse, yaml, numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score, roc_curve, classification_report)
import matplotlib.pyplot as plt
from data import build_loaders, set_seed
from models import create_model

def evaluate(cfg, checkpoint_path: str):
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["out_dir"], exist_ok=True)

    _, _, test_loader, class_names = build_loaders(
        data_root=cfg["data_root"],
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        augment=False,
        workers=cfg.get("workers", 2),
        pin_memory=True,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = create_model(cfg["model"], num_classes=2, pretrained=False, device=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    all_prob, all_pred, all_true = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            p1 = model(x).softmax(1)[:, 1].cpu().numpy()
            all_prob.append(p1); all_pred.append((p1 >= 0.5).astype(int)); all_true.append(y.numpy())

    y_prob = np.concatenate(all_prob)
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)  # sensitivity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp + 1e-12)
    auc  = roc_auc_score(y_true, y_prob)

    metrics = dict(
        accuracy=acc, f1=f1, macro_f1=f1m,
        precision=prec, sensitivity=rec, specificity=spec, roc_auc=auc
    )
    print("TEST METRICS:", {k: round(v,4) for k,v in metrics.items()})
    print("\nReport:\n", classification_report(y_true, y_pred, target_names=class_names))

    # save metrics
    with open(os.path.join(cfg["out_dir"], "metrics_test.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # figures (small)
    fig_dir = os.path.join(cfg["out_dir"], "figures"); os.makedirs(fig_dir, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2,3.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix — Test")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "confusion_matrix.png"), dpi=160)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.2,3.6))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}"); ax.plot([0,1],[0,1],"k--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Test"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "roc_curve.png"), dpi=160)
    plt.close(fig)

    return metrics

def load_yaml(path):
    with open(path, "r") as f: return yaml.safe_load(f)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="configs/efficientnet_b0.yaml")
    p.add_argument("--checkpoint", type=str, required=True, help="path to *_best.pt from training")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    evaluate(cfg, args.checkpoint)
