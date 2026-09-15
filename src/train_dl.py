"""Phase 3 (three EfficientNet-B0 channel variants) and Phase 4 (input
Squeeze-and-Excitation on the full 4-band variant) training script.
"""
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .config import CHANNEL_GROUPS, CHANNEL_NAMES, CLASSES, OUTPUT_DIR, SEED
from .dataset import get_dataloaders
from .models import build_model

OUT = os.path.join(OUTPUT_DIR, "dl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run_epoch(model, loader, loss_fn, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(train_mode):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            loss = loss_fn(out, yb)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            all_preds.append(out.argmax(1).detach().cpu().numpy())
            all_labels.append(yb.detach().cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(labels)
    macro_f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return avg_loss, acc, macro_f1, preds, labels


def run_training(model, train_loader, val_loader, test_loader, name, epochs=100, patience=10, lr=1e-4,
                  save_dir=None, channel_group=None):
    """Core train/val loop with early stopping + final test evaluation. Shared by
    train_one() (fixed train/val/test folders) and train_dl_kfold.py (per-fold
    dataloaders), so the training procedure stays identical between the two.
    Pass save_dir to persist the best checkpoint + epoch history to disk (skipped
    for k-fold, which would otherwise write dozens of large checkpoints)."""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1, _, _ = _run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc, val_f1, _, _ = _run_epoch(model, val_loader, loss_fn, optimizer=None)
        scheduler.step(val_loss)

        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
                         "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1})
        print(f"  epoch {epoch:3d} | train loss {tr_loss:.4f} f1 {tr_f1:.3f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  early stopping at epoch {epoch} (no val-loss improvement for {patience} epochs)")
                break

    elapsed = time.time() - t0
    model.load_state_dict(best_state)

    test_loss, test_acc, test_f1, test_preds, test_labels = _run_epoch(model, test_loader, loss_fn, optimizer=None)
    report = classification_report(test_labels, test_preds, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(test_labels, test_preds).tolist()

    print(f"  [{name}] TEST macro-F1={test_f1:.4f} acc={test_acc:.4f} (best val_loss={best_val_loss:.4f}, {elapsed:.0f}s)")

    channel_weights = None
    if hasattr(model, "last_channel_weights"):
        model.eval()
        weights_acc = []
        with torch.no_grad():
            for xb, _ in test_loader:
                model(xb.to(DEVICE))
                weights_acc.append(model.last_channel_weights.cpu().numpy())
        channel_weights = np.concatenate(weights_acc, axis=0).mean(axis=0).tolist()
        names = [CHANNEL_NAMES[i] for i in CHANNEL_GROUPS[channel_group]] if channel_group else None
        label = ",".join(names) if names else "in order"
        print(f"  learned channel importance ({label}): {channel_weights}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_state, os.path.join(save_dir, f"{name}_best.pt"))
        with open(os.path.join(save_dir, f"{name}_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    return {
        "name": name,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "test_report": report,
        "confusion_matrix": cm,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history),
        "channel_weights": channel_weights,
        "test_predictions": test_preds.tolist(),
        "test_labels": test_labels.tolist(),
    }


def train_one(channel_group, use_se=False, epochs=100, patience=10, batch_size=16, lr=1e-4, seed=SEED):
    torch.manual_seed(seed)
    name = f"{channel_group}{'_SE' if use_se else ''}"
    print(f"\n=== Training {name} ===")

    train_loader, val_loader, test_loader, _ = get_dataloaders(channel_group, batch_size=batch_size)
    model = build_model(channel_group, num_classes=len(CLASSES), pretrained=True, use_se=use_se).to(DEVICE)

    return run_training(model, train_loader, val_loader, test_loader, name,
                         epochs=epochs, patience=patience, lr=lr, save_dir=OUT, channel_group=channel_group)


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {}

    # Phase 3: three EfficientNet-B0 channel variants
    for group in ["visible", "nir", "full"]:
        results[group] = train_one(group, use_se=False)

    # Phase 4: full 4-band variant + input channel Squeeze-and-Excitation
    results["full_SE"] = train_one("full", use_se=True)

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary (test macro-F1) ===")
    for name, r in results.items():
        print(f"  {name:10s}: macro-F1={r['test_macro_f1']:.4f}  acc={r['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
