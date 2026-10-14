"""Inference from the confirmatory-run checkpoints (outputs/dl/confirm_ckpt/*.pt).

A checkpoint stores the best-val-loss weights AND the per-channel mean/std that were used to
normalise that fold's images, so predicting here reproduces training-time inference exactly.
Prediction = softmax averaged over 6 test-time views (identity, h-flip, v-flip, rot90/180/270)
-- the same TTA as src/round2.py.
"""
import numpy as np
import torch

from .config import CHANNEL_GROUPS, CLASSES
from .models import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model = build_model(ckpt["channels"], num_classes=ckpt["n_classes"], pretrained=False, use_se=False,
                        init=ckpt.get("init", "legacy")).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    meta = {k: ckpt[k] for k in ("variant", "seed", "fold", "channels", "source", "n_classes", "mean", "std")}
    return model, meta


def preprocess(cube_uint8, meta):
    """cube_uint8: (H, W, 4) BL/GR/RE/IR, uint8 -> (1, C, H, W) normalised float tensor for this checkpoint."""
    idx = CHANNEL_GROUPS[meta["channels"]]
    arr = np.transpose(cube_uint8.astype(np.float32) / 255.0, (2, 0, 1))[idx]
    mean = np.asarray(meta["mean"], dtype=np.float32)[idx].reshape(-1, 1, 1)
    std = np.asarray(meta["std"], dtype=np.float32)[idx].reshape(-1, 1, 1)
    return torch.from_numpy((arr - mean) / std).unsqueeze(0)


def _views(x):
    return [x, x.flip(3), x.flip(2), torch.rot90(x, 1, (2, 3)), torch.rot90(x, 2, (2, 3)), torch.rot90(x, 3, (2, 3))]


@torch.no_grad()
def predict_probs(model, meta, cube_uint8, tta=True):
    """3-class probabilities (Healthy, Internal_Defect, Surface_Defect) for one cube."""
    x = preprocess(cube_uint8, meta).to(DEVICE)
    views = _views(x) if tta else [x]
    p = torch.stack([torch.softmax(model(v), dim=1) for v in views]).mean(0)[0].cpu().numpy().astype(np.float64)
    if meta["n_classes"] == 5:
        p = np.array([p[0] + p[3] + p[4], p[1], p[2]])
    return p


def soft_vote(prob_list):
    p = np.mean(prob_list, axis=0)
    return p, int(p.argmax())


__all__ = ["CLASSES", "DEVICE", "load_checkpoint", "preprocess", "predict_probs", "soft_vote"]
