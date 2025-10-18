# src/train.py
import os, json, argparse, yaml, numpy as np
import torch, torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC
from data import build_loaders, set_seed
from models import create_model, count_params

def accuracy(logits, y): return (logits.argmax(1) == y).float().mean().item()

def train(cfg):
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # data
    train_loader, val_loader, test_loader, class_names = build_loaders(
        data_root=cfg["data_root"],
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        augment=cfg.get("augment", True),
        workers=cfg.get("workers", 2),
        pin_memory=True,
    )

    # model/opt/sched
    model = create_model(cfg["model"], num_classes=2, pretrained=True, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    auroc = BinaryAUROC().to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    best_auc, best_state, bad = -1.0, None, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}

    print(f"Model: {cfg['model']} | Params (M): {count_params(model):.2f}")

    for ep in range(1, cfg["epochs"] + 1):
        # ---- train ----
        model.train()
        tl, ta, nb = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

            tl += loss.item(); ta += accuracy(logits, y); nb += 1
        sched = sched if sched is None else sched; 
        if sched: sched.step()
        tr_loss, tr_acc = tl / max(nb, 1), ta / max(nb, 1)

        # ---- val ----
        model.eval(); auroc.reset()
        vl, va, vb = 0.0, 0.0, 0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                probs1 = logits.softmax(1)[:, 1]
                auroc.update(probs1, y)
                vloss = F.cross_entropy(logits, y)
                vl += vloss.item(); va += accuracy(logits, y); vb += 1
        va_loss, va_acc, va_auc = vl / max(vb, 1), va / max(vb, 1), float(auroc.compute().item())

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss);   history["val_acc"].append(va_acc)
        history["val_auc"].append(va_auc)

        print(f"Epoch {ep:02d}/{cfg['epochs']} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} | val AUC {va_auc:.4f}")

        # early stop on AUC
        if va_auc > best_auc:
            best_auc, best_state, bad = va_auc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad > cfg["patience"]:
                print("Early stopping triggered."); break

    # save
    model.load_state_dict(best_state)
    ckpt_path = os.path.join(cfg["out_dir"], f"{cfg['model']}_best.pt")
    torch.save({"state_dict": model.state_dict(), "best_val_auc": best_auc, "history": history}, ckpt_path)
    with open(os.path.join(cfg["out_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("Saved:", ckpt_path, "| Best val ROC-AUC:", best_auc)
    return ckpt_path, history, class_names

def load_yaml(path):
    with open(path, "r") as f: return yaml.safe_load(f)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="configs/efficientnet_b0.yaml")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    train(cfg)
