"""
fog_lstm.py
-----------
Bidirectional LSTM for Freezing-of-Gait (FOG) prediction.

Input:  (batch=1, seq_len=30, features=5)
           features: hr_norm, temp_norm, activity, proximity, is_freezing
Output: (state_logits[5], freeze_prob, neuro_load)

Usage:
    from app.services.fog_lstm import FogLSTM, train_lstm, load_lstm
    model = load_lstm()
    logits, freeze_prob, neuro_load = model(window_tensor)
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN   = 30      # frames of history fed to LSTM  (~1.5 s at 20 Hz)
N_FEAT    = 5       # hr, temp, activity, proximity, is_freezing
N_CLASSES = 5       # NOMINAL, STABLE, ELEVATED, SPATIAL, FREEZE
HIDDEN    = 64
LAYERS    = 2
CKPT_NAME = "fog_lstm.pth"

STATE_MAP = {"NOMINAL": 0, "STABLE": 1, "ELEVATED": 2, "SPATIAL": 3, "FREEZE": 4}
IDX_MAP   = {v: k for k, v in STATE_MAP.items()}

_basedir = os.path.dirname(__file__)

# ── Model ─────────────────────────────────────────────────────────────────────

class FogLSTM(nn.Module):
    """Bidirectional 2-layer LSTM with two output heads."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=N_FEAT,
            hidden_size=HIDDEN,
            num_layers=LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        feat_dim = HIDDEN * 2  # bidirectional doubles hidden size

        # Classification head (patient state)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, N_CLASSES),
        )

        # Regression heads
        self.freeze_head = nn.Sequential(
            nn.Linear(feat_dim, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )
        self.neuro_head = nn.Sequential(
            nn.Linear(feat_dim, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq, features)
        out, _ = self.lstm(x)          # (batch, seq, hidden*2)
        ctx    = out[:, -1, :]         # use last timestep context
        return (
            self.classifier(ctx),      # (batch, N_CLASSES)
            self.freeze_head(ctx).squeeze(-1),   # (batch,)
            self.neuro_head(ctx).squeeze(-1),    # (batch,)
        )


# ── Data generation ───────────────────────────────────────────────────────────

def _generate_sequences(n_sequences: int = 8000, seed: int = 0):
    """
    Generate synthetic FOG time-series sequences.
    Each sequence is 30 frames long.  The key insight is that
    a FOG event is *preceded* by rising proximity and dropping activity.
    """
    rng = np.random.default_rng(seed)
    Xs, y_states, y_freeze, y_neuro = [], [], [], []

    for _ in range(n_sequences):
        # Pick a target scenario
        scenario = rng.choice(["NOMINAL", "STABLE", "ELEVATED", "SPATIAL", "FREEZE"],
                               p=[0.30, 0.25, 0.20, 0.15, 0.10])

        seq = []
        for t in range(SEQ_LEN):
            progress = t / SEQ_LEN  # 0 → 1

            if scenario == "FREEZE":
                # Progressive deterioration leading to freeze
                hr   = rng.normal(85 + progress * 20, 5)
                temp = rng.normal(36.9 + progress * 0.3, 0.1)
                act  = max(0.0, 0.8 - progress * 0.9 + rng.normal(0, 0.05))
                prox = min(1.0, progress * 0.95 + rng.normal(0, 0.05))
                frz  = 1.0 if progress > 0.75 else 0.0

            elif scenario == "SPATIAL":
                hr   = rng.normal(80, 8)
                temp = rng.normal(36.8, 0.15)
                act  = rng.normal(0.5, 0.1)
                prox = min(1.0, 0.7 + rng.normal(0, 0.08))
                frz  = 0.0

            elif scenario == "ELEVATED":
                hr   = rng.normal(105, 10)
                temp = rng.normal(37.1, 0.2)
                act  = rng.normal(0.3, 0.12)
                prox = rng.uniform(0, 0.5)
                frz  = 0.0

            elif scenario == "STABLE":
                hr   = rng.normal(72, 5)
                temp = rng.normal(36.6, 0.1)
                act  = rng.normal(0.7, 0.1)
                prox = rng.uniform(0, 0.3)
                frz  = 0.0

            else:  # NOMINAL
                hr   = rng.normal(68, 7)
                temp = rng.normal(36.6, 0.12)
                act  = rng.exponential(0.4)
                prox = rng.uniform(0, 0.4)
                frz  = 0.0

            # Normalise to ~[0, 1]
            hr_n   = (np.clip(hr,   40, 180) - 60) / 120.0
            temp_n = (np.clip(temp, 35,  40) - 36.0) / 3.0
            act_n  = float(np.clip(act,  0,   2.0)) / 2.0
            seq.append([hr_n, temp_n, act_n, float(np.clip(prox, 0, 1)), float(frz)])

        Xs.append(seq)
        y_states.append(STATE_MAP[scenario])

        # Freeze probability in next 3 seconds = 1 if scenario is FREEZE, scaled otherwise
        y_freeze.append(1.0 if scenario == "FREEZE" else
                        (0.6 if scenario == "SPATIAL" else 0.1))
        y_neuro.append( 0.85 if scenario == "FREEZE" else
                        (0.65 if scenario == "SPATIAL" else
                         (0.50 if scenario == "ELEVATED" else 0.15)))

    X  = torch.tensor(Xs, dtype=torch.float32)
    ys = torch.tensor(y_states, dtype=torch.long)
    yf = torch.tensor(y_freeze, dtype=torch.float32)
    yn = torch.tensor(y_neuro,  dtype=torch.float32)
    return X, ys, yf, yn


# ── Training ──────────────────────────────────────────────────────────────────

def train_lstm(epochs: int = 30, batch_size: int = 64, verbose: bool = True):
    """Train the FogLSTM and save a checkpoint."""
    print("Generating FOG time-series training data …")
    X, ys, yf, yn = _generate_sequences(n_sequences=10000)

    n     = len(X)
    split = int(n * 0.85)
    perm  = torch.randperm(n)
    tr, va = perm[:split], perm[split:]

    train_ds = TensorDataset(X[tr], ys[tr], yf[tr], yn[tr])
    val_ds   = TensorDataset(X[va], ys[va], yf[va], yn[va])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    model  = FogLSTM()
    opt    = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce     = nn.CrossEntropyLoss()
    bce    = nn.BCELoss()

    print("Training FogLSTM …")
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        for xb, ysb, yfb, ynb in train_dl:
            logits, fp, nl = model(xb)
            loss = ce(logits, ysb) + bce(fp, yfb) + bce(nl, ynb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            t_loss += loss.item()
        sched.step()

        model.eval()
        v_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, ysb, yfb, ynb in val_dl:
                logits, fp, nl = model(xb)
                v_loss  += (ce(logits, ysb) + bce(fp, yfb) + bce(nl, ynb)).item()
                correct += (logits.argmax(1) == ysb).sum().item()

        acc = correct / len(val_ds) * 100
        if verbose and epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  train={t_loss/len(train_dl):.4f}"
                  f"  val={v_loss/len(val_dl):.4f}  acc={acc:.1f}%")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), os.path.join(_basedir, CKPT_NAME))

    print(f"FogLSTM training complete. Best val loss: {best_val_loss:.4f}")
    return model


# ── Inference helpers ─────────────────────────────────────────────────────────

def load_lstm() -> FogLSTM | None:
    """Load the saved checkpoint if it exists."""
    path = os.path.join(_basedir, CKPT_NAME)
    if not os.path.exists(path):
        return None
    model = FogLSTM()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def predict(model: FogLSTM, window: list[list[float]]):
    """
    Run inference on a list of SEQ_LEN feature vectors.
    Returns (state_str, freeze_prob, neuro_load).
    """
    x = torch.tensor([window], dtype=torch.float32)  # (1, seq, feat)
    with torch.no_grad():
        logits, fp, nl = model(x)
    state_idx  = int(logits.argmax(1).item())
    return IDX_MAP[state_idx], float(fp.item()), float(nl.item())


if __name__ == "__main__":
    train_lstm()
