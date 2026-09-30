"""
Lightweight multi-label classification head over frozen LLaVA-Med CLIP patch tokens (DR-016).

Architecture: per-token LayerNorm -> Linear(1024, hidden) -> GELU, mean-pool over the 24x24
patch grid, dropout, Linear(hidden, 14). Trained on train-split CheXbert reference labels only;
decision thresholds are tuned on val.
"""

from typing import Dict

import numpy as np
import torch
from torch import nn


class PatchTokenHead(nn.Module):
    def __init__(self, in_dim: int = 1024, hidden: int = 512, n_classes: int = 14, dropout: float = 0.2):
        super().__init__()
        self.token_mlp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU())
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, 576, in_dim) -> logits (B, n_classes)."""
        return self.classifier(self.dropout(self.token_mlp(tokens).mean(dim=1)))


def train_head(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
               epochs: int = 30, lr: float = 1e-3, weight_decay: float = 1e-2, batch_size: int = 64,
               seed: int = 42, device: str = None) -> Dict:
    """Trains with class-balanced BCE; keeps the epoch with the best val macro AUROC.

    Features may be np.memmap (float16); batches are cast to float32 on the fly.
    """
    from sklearn.metrics import roc_auc_score

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n_classes = train_y.shape[1]
    head = PatchTokenHead(in_dim=train_x.shape[2], n_classes=n_classes).to(device)
    pos = train_y.sum(axis=0)
    pos_weight = torch.tensor(np.clip((len(train_y) - pos) / np.maximum(pos, 1), 1.0, 50.0),
                              dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def predict(x):
        head.eval()
        probs = []
        with torch.no_grad():
            for s in range(0, len(x), 256):
                xb = torch.as_tensor(np.asarray(x[s: s + 256], dtype=np.float32), device=device)
                probs.append(torch.sigmoid(head(xb)).cpu().numpy())
        return np.concatenate(probs)

    def macro_auc(y, p):
        aucs = [roc_auc_score(y[:, c], p[:, c]) for c in range(n_classes) if 0 < y[:, c].sum() < len(y)]
        return float(np.mean(aucs)) if aucs else float("nan")

    history, best = [], {"val_macro_auroc": -1.0}
    for epoch in range(epochs):
        head.train()
        order = rng.permutation(len(train_x))
        total = 0.0
        for s in range(0, len(order), batch_size):
            idx = np.sort(order[s: s + batch_size])  # sorted reads are much faster on memmaps
            xb = torch.as_tensor(np.asarray(train_x[idx], dtype=np.float32), device=device)
            yb = torch.as_tensor(train_y[idx], dtype=torch.float32, device=device)
            loss = loss_fn(head(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        val_auc = macro_auc(val_y, predict(val_x))
        history.append({"epoch": epoch + 1, "train_loss": total / len(train_x), "val_macro_auroc": val_auc})
        if val_auc > best["val_macro_auroc"]:
            best = {"val_macro_auroc": val_auc, "epoch": epoch + 1,
                    "state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}

    head.load_state_dict(best["state_dict"])
    return {"head": head.eval(), "best_epoch": best["epoch"], "best_val_macro_auroc": best["val_macro_auroc"],
            "history": history, "predict": predict}


def tune_thresholds(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Per-class threshold maximising F1 on val; classes without positives default to 0.5."""
    grid = np.linspace(0.05, 0.95, 91)
    thresholds = np.full(y_true.shape[1], 0.5)
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            continue
        pred = probs[:, c][None, :] >= grid[:, None]
        tp = (pred & (y_true[:, c] == 1)).sum(axis=1)
        f1 = 2 * tp / np.maximum(pred.sum(axis=1) + y_true[:, c].sum(), 1)
        thresholds[c] = grid[int(np.argmax(f1))]
    return thresholds
