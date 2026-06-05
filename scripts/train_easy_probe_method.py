"""Train and evaluate one easy-probe method end-to-end.

Methods: random, dense_linear, sae_positive, sae_mixed, sae_contrastive.

Trains the representation (if any) on PRM800K cached hidden states, trains a
fresh linear probe on probe_train_40k, selects a threshold on val_1k using the
10-point grid 0.1..1.0, then scores ProcessBench-GSM8K steps and writes the
official ProcessBench first-error metrics.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.repr.objectives import ranking_loss, triplet_loss  # noqa: E402

# Sprint 1 methods + Sprint 2 fork-based representation-shaping methods.
#   ae               : dense autoencoder, recon only on mixed_train_40k (no pairs)
#   {ae,sae}_recon   : MATCHED control - recon[+L1] on the SAME fork items as the
#                      objective methods, objective term OFF (obj_weight=0). The
#                      only difference from *_rank/*_triplet is the missing term.
#   {ae,sae}_rank    : recon[+L1] + pairwise ranking on fork pos/neg siblings
#   {ae,sae}_triplet : recon[+L1] + triplet (anchor=prefix, pos/neg continuations)
METHODS = [
    "random", "dense_linear", "sae_positive", "sae_mixed", "sae_contrastive",
    "ae", "ae_recon", "sae_recon",
    "ae_rank", "sae_rank", "ae_triplet", "sae_triplet",
]
FORK_METHODS = (
    "ae_recon", "sae_recon",
    "ae_rank", "sae_rank", "ae_triplet", "sae_triplet",
)
THRESHOLD_GRID = [round(0.1 * i, 1) for i in range(1, 11)]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_npy_pair(cache_dir: Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    h = np.load(cache_dir / f"{stem}_h.npy")
    y = np.load(cache_dir / f"{stem}_y.npy")
    if h.shape[0] != y.shape[0]:
        raise ValueError(f"{stem}: h/y row mismatch {h.shape[0]} vs {y.shape[0]}")
    y = y.astype(np.int64)
    unique = np.unique(y)
    if not set(unique.tolist()).issubset({0, 1}):
        raise ValueError(f"{stem}: y must contain only {{0, 1}}, got {unique.tolist()}")
    return h.astype(np.float32), y


def require_pb_cache(
    pb_cache_dir: Path | None = None,
    pb_h: Path | None = None,
    pb_meta: Path | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Load ProcessBench hidden states + meta.

    Either pass --pb_h/--pb_meta explicitly, or pass --pb_cache_dir and we
    will look for the legacy pb_gsm8k_step_* names, falling back to generic
    pb_step_* names emitted by the new multi-subset encoder.
    """
    if pb_h is not None and pb_meta is not None:
        h_path = pb_h
        meta_path = pb_meta
    else:
        if pb_cache_dir is None:
            raise ValueError("Must provide pb_cache_dir or (pb_h, pb_meta)")
        legacy_h = pb_cache_dir / "pb_gsm8k_step_h.npy"
        legacy_meta = pb_cache_dir / "pb_gsm8k_step_meta.jsonl"
        new_h = pb_cache_dir / "pb_step_h.npy"
        new_meta = pb_cache_dir / "pb_step_meta.jsonl"
        if legacy_h.exists() and legacy_meta.exists():
            h_path, meta_path = legacy_h, legacy_meta
        else:
            h_path, meta_path = new_h, new_meta
    if not h_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"ProcessBench cached hidden states missing: expected\n"
            f"  {h_path}\n  {meta_path}\n"
            "Cannot evaluate. Re-run the ProcessBench encoder."
        )
    h = np.load(h_path).astype(np.float32)
    meta = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
    if len(meta) != h.shape[0]:
        raise ValueError(
            f"ProcessBench meta rows ({len(meta)}) != h rows ({h.shape[0]})"
        )
    # Each ProcessBench trace must have a unique label across its rows.
    seen: dict[str, int] = {}
    for row in meta:
        tid = row["id"]
        lbl = int(row["label"])
        if tid in seen and seen[tid] != lbl:
            raise ValueError(
                f"ProcessBench trace {tid} has inconsistent labels: "
                f"{seen[tid]} vs {lbl}"
            )
        seen[tid] = lbl
    return h, meta


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SAE(nn.Module):
    """Single-layer SAE: z = ReLU(W_enc h + b_enc); h_hat = W_dec z + b_dec."""

    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(hidden_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, hidden_dim, bias=True)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(h))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(h)
        h_hat = self.decoder(z)
        return z, h_hat


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------

def iter_minibatches(
    n: int, batch_size: int, rng: np.random.Generator, shuffle: bool = True
) -> list[np.ndarray]:
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    return [idx[i : i + batch_size] for i in range(0, n, batch_size)]


def train_sae(
    h_train: np.ndarray,
    y_train: np.ndarray | None,
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    l1_weight: float,
    bce_weight: float,
    contrastive: bool,
    device: torch.device,
    seed: int,
) -> tuple[SAE, dict]:
    rng = np.random.default_rng(seed)
    sae = SAE(hidden_dim, latent_dim).to(device)
    aux: nn.Linear | None = None
    params = list(sae.parameters())
    if contrastive:
        aux = nn.Linear(latent_dim, 1).to(device)
        params += list(aux.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    h_tensor = torch.from_numpy(h_train).to(device)
    y_tensor = (
        torch.from_numpy(y_train.astype(np.float32)).to(device)
        if y_train is not None
        else None
    )

    n = h_tensor.shape[0]
    final_mse = float("nan")
    final_l1 = float("nan")
    final_bce = float("nan")
    for epoch in range(epochs):
        sae.train()
        if aux is not None:
            aux.train()
        ep_mse, ep_l1, ep_bce, n_batches = 0.0, 0.0, 0.0, 0
        for batch_idx in iter_minibatches(n, batch_size, rng):
            ib = torch.from_numpy(batch_idx).to(device)
            h_b = h_tensor.index_select(0, ib)
            z, h_hat = sae(h_b)
            mse = F.mse_loss(h_hat, h_b)
            l1 = z.abs().mean()
            loss = mse + l1_weight * l1
            bce_val = torch.tensor(0.0, device=device)
            if contrastive and aux is not None and y_tensor is not None:
                y_b = y_tensor.index_select(0, ib)
                logit = aux(z).squeeze(-1)
                bce_val = F.binary_cross_entropy_with_logits(logit, y_b)
                loss = loss + bce_weight * bce_val
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_mse += mse.item()
            ep_l1 += l1.item()
            ep_bce += bce_val.item()
            n_batches += 1
        final_mse = ep_mse / max(n_batches, 1)
        final_l1 = ep_l1 / max(n_batches, 1)
        final_bce = ep_bce / max(n_batches, 1)
    stats = {
        "final_reconstruction_mse": final_mse,
        "final_l1_mean": final_l1,
        "final_aux_bce": final_bce if contrastive else None,
    }
    return sae, stats


def load_fork_pairs(
    items_meta_path: Path, pairs_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map fork pair UIDs to row indices into the encoded items array.

    Returns (anchor_idx, pos_idx, neg_idx), each shape (n_pairs,).
    """
    uid_to_row: dict[str, int] = {}
    for line in items_meta_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        uid_to_row[row["item_uid"]] = int(row["row"])

    anchor, pos, neg = [], [], []
    for line in pairs_path.read_text().splitlines():
        if not line.strip():
            continue
        pr = json.loads(line)
        try:
            anchor.append(uid_to_row[pr["anchor_uid"]])
            pos.append(uid_to_row[pr["positive_uid"]])
            neg.append(uid_to_row[pr["negative_uid"]])
        except KeyError as e:
            raise ValueError(
                f"Pair references item uid {e} absent from items meta "
                f"{items_meta_path}. Re-encode items and pairs from the same build."
            )
    if not pos:
        raise ValueError(f"No pairs loaded from {pairs_path}.")
    return np.array(anchor), np.array(pos), np.array(neg)


def train_repr_with_pairs(
    items_h: np.ndarray,
    anchor_idx: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    l1_weight: float,
    objective: str,
    obj_weight: float,
    rank_kind: str,
    rank_margin: float,
    triplet_metric: str,
    triplet_margin: float,
    device: torch.device,
    seed: int,
) -> tuple[SAE, dict]:
    """Train an encoder with reconstruction[+L1] plus a fork preference term.

    objective: "rank" (scalar head, score_pos > score_neg) or "triplet"
    (latent-space anchor->pos pull, anchor->neg push). The ranking head is
    auxiliary and discarded; only the SAE/AE encoder+decoder are saved, so the
    artifact stays format-compatible with the ProcessBench eval pipeline.
    """
    if objective not in ("rank", "triplet", "none"):
        raise ValueError(f"objective must be rank|triplet|none, got {objective!r}")

    rng = np.random.default_rng(seed)
    sae = SAE(hidden_dim, latent_dim).to(device)
    head: nn.Linear | None = None
    params = list(sae.parameters())
    if objective == "rank":
        head = nn.Linear(latent_dim, 1).to(device)
        params += list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    h_t = torch.from_numpy(items_h).to(device)
    a_t = torch.from_numpy(anchor_idx).to(device)
    p_t = torch.from_numpy(pos_idx).to(device)
    n_t = torch.from_numpy(neg_idx).to(device)
    n_pairs = pos_idx.shape[0]

    def pair_metrics(za, zp, zn, s_pos, s_neg):
        """Detached diagnostics: did the objective actually order the pair?

        pair_acc = fraction of pairs ranked/embedded in the correct order.
        margin_sat = fraction satisfying the configured margin.
        Returns (pair_acc, margin_sat) as floats, or (nan, nan) for objective=none.
        """
        with torch.no_grad():
            if objective == "rank":
                diff = (s_pos - s_neg).detach()
                return (diff > 0).float().mean().item(), (diff >= rank_margin).float().mean().item()
            if objective == "triplet":
                if triplet_metric == "l2":
                    d_pos = (za - zp).pow(2).sum(-1)
                    d_neg = (za - zn).pow(2).sum(-1)
                else:
                    d_pos = 1.0 - F.cosine_similarity(za, zp, dim=-1)
                    d_neg = 1.0 - F.cosine_similarity(za, zn, dim=-1)
                gap = (d_neg - d_pos).detach()
                return (gap > 0).float().mean().item(), (gap >= triplet_margin).float().mean().item()
            return float("nan"), float("nan")

    history: list[dict] = []
    final = {"recon": float("nan"), "l1": float("nan"), "obj": float("nan"),
             "pair_acc": float("nan"), "margin_sat": float("nan")}
    for epoch in range(epochs):
        sae.train()
        if head is not None:
            head.train()
        ep_recon = ep_l1 = ep_obj = ep_acc = ep_margin = 0.0
        nb = 0
        for batch in iter_minibatches(n_pairs, batch_size, rng):
            ib = torch.from_numpy(batch).to(device)
            ha = h_t.index_select(0, a_t.index_select(0, ib))
            hp = h_t.index_select(0, p_t.index_select(0, ib))
            hn = h_t.index_select(0, n_t.index_select(0, ib))
            za, ha_hat = sae(ha)
            zp, hp_hat = sae(hp)
            zn, hn_hat = sae(hn)
            recon = (
                F.mse_loss(ha_hat, ha)
                + F.mse_loss(hp_hat, hp)
                + F.mse_loss(hn_hat, hn)
            ) / 3.0
            l1 = (za.abs().mean() + zp.abs().mean() + zn.abs().mean()) / 3.0
            loss = recon + l1_weight * l1
            s_pos = s_neg = None
            if objective == "rank":
                s_pos = head(zp).squeeze(-1)
                s_neg = head(zn).squeeze(-1)
                obj = ranking_loss(s_pos, s_neg, kind=rank_kind, margin=rank_margin)
                loss = loss + obj_weight * obj
            elif objective == "triplet":
                obj = triplet_loss(za, zp, zn, metric=triplet_metric, margin=triplet_margin)
                loss = loss + obj_weight * obj
            else:  # none (recon-only control)
                obj = torch.zeros((), device=device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            acc, margin = pair_metrics(za, zp, zn, s_pos, s_neg)
            ep_recon += recon.item()
            ep_l1 += l1.item()
            ep_obj += obj.item()
            ep_acc += acc
            ep_margin += margin
            nb += 1
        final = {
            "recon": ep_recon / max(nb, 1),
            "l1": ep_l1 / max(nb, 1),
            "obj": ep_obj / max(nb, 1),
            "pair_acc": ep_acc / max(nb, 1),
            "margin_sat": ep_margin / max(nb, 1),
        }
        history.append({"epoch": epoch, **final})

    stats = {
        "final_reconstruction_mse": final["recon"],
        "final_l1_mean": final["l1"],
        "final_aux_bce": None,
        "final_objective_loss": final["obj"],
        "final_pair_accuracy": final["pair_acc"],
        "final_margin_satisfaction": final["margin_sat"],
        "objective": objective,
        "obj_weight": obj_weight,
        "n_pairs": int(n_pairs),
        "history": history,
    }
    return sae, stats


def train_linear_probe(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device: torch.device,
    seed: int,
) -> LinearProbe:
    rng = np.random.default_rng(seed)
    in_dim = z_train.shape[1]
    probe = LinearProbe(in_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.0)

    z_train_t = torch.from_numpy(z_train).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    z_val_t = torch.from_numpy(z_val).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(device)

    n = z_train_t.shape[0]
    best_val_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    bad_epochs = 0

    for epoch in range(epochs):
        probe.train()
        for batch_idx in iter_minibatches(n, batch_size, rng):
            ib = torch.from_numpy(batch_idx).to(device)
            zb = z_train_t.index_select(0, ib)
            yb = y_train_t.index_select(0, ib)
            logit = probe(zb)
            loss = F.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            val_logit = probe(z_val_t)
            val_loss = F.binary_cross_entropy_with_logits(val_logit, y_val_t).item()
        if val_loss + 1e-8 < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    probe.load_state_dict(best_state)
    return probe


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def encode_with_sae(sae: SAE, h: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    sae.eval()
    out = []
    with torch.no_grad():
        for i in range(0, h.shape[0], batch_size):
            chunk = torch.from_numpy(h[i : i + batch_size]).to(device)
            z = sae.encode(chunk).cpu().numpy()
            out.append(z)
    return np.concatenate(out, axis=0)


def probe_scores(probe: LinearProbe, z: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    probe.eval()
    out = []
    with torch.no_grad():
        for i in range(0, z.shape[0], batch_size):
            chunk = torch.from_numpy(z[i : i + batch_size]).to(device)
            logit = probe(chunk)
            out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def select_threshold(
    scores: np.ndarray, y: np.ndarray, grid: list[float] | None = None
) -> tuple[float, float, float]:
    grid = grid if grid is not None else THRESHOLD_GRID
    best_t = grid[0]
    best_bacc = -1.0
    best_f1 = 0.0
    for t in grid:
        pred = (scores > t).astype(np.int64)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        bacc = 0.5 * (tpr + tnr)
        prec = tp / max(tp + fp, 1)
        rec = tpr
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        if bacc > best_bacc + 1e-12:
            best_bacc = bacc
            best_t = t
            best_f1 = f1
    return best_t, best_bacc, best_f1


# ---------------------------------------------------------------------------
# ProcessBench evaluation
# ---------------------------------------------------------------------------

def evaluate_processbench(
    scores: np.ndarray, meta: list[dict], threshold: float
) -> tuple[list[dict], dict]:
    groups: dict[str, list[tuple[int, float]]] = {}
    labels: dict[str, int] = {}
    n_steps_map: dict[str, int] = {}
    for i, row in enumerate(meta):
        tid = row["id"]
        groups.setdefault(tid, []).append((int(row["step_idx"]), float(scores[i])))
        labels[tid] = int(row["label"])
        n_steps_map[tid] = int(row["n_steps"])

    rows: list[dict] = []
    n_error = n_correct = 0
    acc_error_hits = 0
    acc_correct_hits = 0
    exact_all = 0
    for tid, items in groups.items():
        items.sort(key=lambda x: x[0])
        step_scores = [s for _, s in items]
        pred = -1
        for t_idx, s in items:
            if s > threshold:
                pred = t_idx
                break
        label = labels[tid]
        rows.append(
            {
                "id": tid,
                "label": label,
                "n_steps": n_steps_map[tid],
                "scores": step_scores,
                "threshold": threshold,
                "prediction": pred,
            }
        )
        if label == -1:
            n_correct += 1
            if pred == -1:
                acc_correct_hits += 1
        else:
            n_error += 1
            if pred == label:
                acc_error_hits += 1
        if pred == label:
            exact_all += 1

    acc_error = acc_error_hits / max(n_error, 1) if n_error else 0.0
    acc_correct = acc_correct_hits / max(n_correct, 1) if n_correct else 0.0
    denom = acc_error + acc_correct
    f1_pb = (2 * acc_error * acc_correct / denom) if denom > 0 else 0.0
    exact_match_all = exact_all / max(len(groups), 1)

    metrics = {
        "n_traces": len(groups),
        "n_error_traces": n_error,
        "n_correct_traces": n_correct,
        "Acc_error": acc_error,
        "Acc_correct": acc_correct,
        "F1_PB": f1_pb,
        "Exact_match_all": exact_match_all,
    }
    return rows, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--cache_dir", required=True, type=Path)
    p.add_argument("--pb_cache_dir", required=False, type=Path, default=None,
                   help="Legacy: directory with pb_gsm8k_step_* (or pb_step_*). "
                        "If omitted, use --pb_h/--pb_meta or --pb_specs.")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs_sae", type=int, default=20)
    p.add_argument("--epochs_probe", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr_sae", type=float, default=1e-3)
    p.add_argument("--lr_probe", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--l1_weight", type=float, default=1e-4)
    p.add_argument("--bce_weight", type=float, default=0.1)
    p.add_argument("--latent_dim", type=int, default=None,
                   help="Default: hidden_dim (no overcomplete unless explicitly set).")
    p.add_argument("--early_stopping_patience", type=int, default=5)
    # ---- Sprint 2: fork-based representation shaping -------------------
    p.add_argument("--fork_items_h", type=Path, default=None,
                   help="Encoded fork items .npy (required for *_rank / *_triplet).")
    p.add_argument("--fork_items_meta", type=Path, default=None,
                   help="Fork items meta .jsonl mapping item_uid -> row.")
    p.add_argument("--fork_pairs", type=Path, default=None,
                   help="Fork pairs .jsonl (anchor/positive/negative uids).")
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Truncate fork pairs to the first N (sanity/smoke runs).")
    p.add_argument("--obj_weight", type=float, default=1.0,
                   help="Weight on the ranking/triplet term added to recon[+L1].")
    p.add_argument("--rank_kind", choices=["logistic", "margin"], default="logistic")
    p.add_argument("--rank_margin", type=float, default=1.0)
    p.add_argument("--triplet_metric", choices=["l2", "cosine"], default="l2")
    p.add_argument("--triplet_margin", type=float, default=1.0)
    # ---- Generalized cache stems --------------------------------------
    p.add_argument("--probe_train_stem", type=str, default="probe_train_40k",
                   help="Stem under --cache_dir loaded as probe training data.")
    p.add_argument("--val_stem", type=str, default="val_1k",
                   help="Stem under --cache_dir loaded as validation data.")
    # ---- Single explicit PB target ------------------------------------
    p.add_argument("--pb_h", type=Path, default=None,
                   help="Explicit path to ProcessBench hidden states .npy.")
    p.add_argument("--pb_meta", type=Path, default=None,
                   help="Explicit path to ProcessBench step meta .jsonl.")
    p.add_argument("--pb_name", type=str, default="gsm8k",
                   help="Tag for output filenames (pb_step_scores_<pb_name>.jsonl etc).")
    # ---- Multiple PB targets in one run -------------------------------
    p.add_argument(
        "--pb_specs", type=str, nargs="+", default=None,
        help="Optional list of 'name:h_path:meta_path' triples. When set, the "
             "trained probe is evaluated against every PB target sequentially.",
    )
    # ---- Threshold grid -----------------------------------------------
    p.add_argument(
        "--threshold_grid", type=str, default=None,
        help="Either a float step size in (0,1) (e.g. 0.001) or a comma-"
             "separated list of explicit thresholds (e.g. '0.3,0.5,0.7'). "
             "Default: 0.1..1.0 by 0.1 (legacy).",
    )
    # ---- Sanity / compatibility ---------------------------------------
    p.add_argument("--skip_size_asserts", action="store_true",
                   help="Skip the legacy 40k/1k size assertions (required for "
                        "any non-default --probe_train_stem / --val_stem).")
    return p.parse_args()


def resolve_threshold_grid(arg: str | None) -> list[float]:
    if arg is None:
        return THRESHOLD_GRID
    if "," in arg:
        return [float(t) for t in arg.split(",") if t.strip()]
    step = float(arg)
    if step <= 0 or step >= 1:
        raise SystemExit(f"--threshold_grid step must be in (0,1), got {step}")
    n = int(round(1.0 / step))
    return [round(step * i, 6) for i in range(1, n)]


def parse_pb_specs(specs: list[str]) -> list[tuple[str, Path, Path]]:
    out: list[tuple[str, Path, Path]] = []
    for s in specs:
        parts = s.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--pb_specs entry must be name:h:meta, got {s!r}")
        name, h, meta = parts
        out.append((name, Path(h), Path(meta)))
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else ""

    # Persist config
    cfg = vars(args).copy()
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()}
    cfg["device"] = device.type
    cfg["gpu_name"] = gpu_name
    (args.out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))

    # Load PRM800K data
    probe_train_h, probe_train_y = load_npy_pair(args.cache_dir, args.probe_train_stem)
    val_h, val_y = load_npy_pair(args.cache_dir, args.val_stem)
    hidden_dim = probe_train_h.shape[1]
    latent_dim = args.latent_dim if args.latent_dim is not None else hidden_dim

    legacy_default = (args.probe_train_stem == "probe_train_40k" and args.val_stem == "val_1k")
    if legacy_default and not args.skip_size_asserts:
        assert probe_train_h.shape[0] == 40000, (
            f"probe_train_40k must have 40000 rows, got {probe_train_h.shape[0]}"
        )
        assert val_h.shape[0] == 1000, (
            f"val_1k must have 1000 rows, got {val_h.shape[0]}"
        )

    # Resolve ProcessBench targets
    pb_targets: list[tuple[str, np.ndarray, list[dict]]] = []
    if args.pb_specs:
        for name, h_path, meta_path in parse_pb_specs(args.pb_specs):
            h, meta = require_pb_cache(pb_h=h_path, pb_meta=meta_path)
            assert h.shape[1] == hidden_dim, (
                f"PB[{name}] hidden_dim ({h.shape[1]}) != PRM800K ({hidden_dim})"
            )
            pb_targets.append((name, h, meta))
    elif args.pb_h is not None and args.pb_meta is not None:
        h, meta = require_pb_cache(pb_h=args.pb_h, pb_meta=args.pb_meta)
        assert h.shape[1] == hidden_dim
        pb_targets.append((args.pb_name, h, meta))
    elif args.pb_cache_dir is not None:
        h, meta = require_pb_cache(pb_cache_dir=args.pb_cache_dir)
        assert h.shape[1] == hidden_dim
        pb_targets.append((args.pb_name, h, meta))
    else:
        raise SystemExit("Provide one of: --pb_specs, --pb_h+--pb_meta, or --pb_cache_dir")

    threshold_grid = resolve_threshold_grid(args.threshold_grid)

    method = args.method

    representation_train_time = 0.0
    probe_train_time = 0.0
    rep_train_n = 0
    sae_stats = {
        "final_reconstruction_mse": None,
        "final_l1_mean": None,
        "final_aux_bce": None,
    }

    total_t0 = time.time()

    # ---- Representation step ----------------------------------------------
    sae: SAE | None = None
    if method in ("sae_positive", "sae_mixed", "sae_contrastive"):
        if method == "sae_positive":
            rep_h, rep_y = load_npy_pair(args.cache_dir, "pos_base_20k")
            contrastive = False
            rep_y_for_train = None
        elif method == "sae_mixed":
            rep_h, rep_y = load_npy_pair(args.cache_dir, "mixed_train_40k")
            contrastive = False
            rep_y_for_train = None
        else:  # sae_contrastive
            rep_h, rep_y = load_npy_pair(args.cache_dir, "mixed_train_40k")
            contrastive = True
            rep_y_for_train = rep_y
        rep_train_n = rep_h.shape[0]
        t0 = time.time()
        sae, sae_stats = train_sae(
            h_train=rep_h,
            y_train=rep_y_for_train,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            epochs=args.epochs_sae,
            batch_size=args.batch_size,
            lr=args.lr_sae,
            l1_weight=args.l1_weight,
            bce_weight=args.bce_weight,
            contrastive=contrastive,
            device=device,
            seed=args.seed,
        )
        representation_train_time = time.time() - t0
        torch.save(sae.state_dict(), args.out_dir / "representation.pt")
        for p in sae.parameters():
            p.requires_grad_(False)
        sae.eval()
    elif method == "ae":
        # Dense autoencoder: reconstruction only (no sparsity, no labels).
        rep_h, _ = load_npy_pair(args.cache_dir, "mixed_train_40k")
        rep_train_n = rep_h.shape[0]
        t0 = time.time()
        sae, sae_stats = train_sae(
            h_train=rep_h, y_train=None,
            hidden_dim=hidden_dim, latent_dim=latent_dim,
            epochs=args.epochs_sae, batch_size=args.batch_size, lr=args.lr_sae,
            l1_weight=0.0, bce_weight=0.0, contrastive=False,
            device=device, seed=args.seed,
        )
        representation_train_time = time.time() - t0
        torch.save(sae.state_dict(), args.out_dir / "representation.pt")
        for p in sae.parameters():
            p.requires_grad_(False)
        sae.eval()
    elif method in FORK_METHODS:
        if not (args.fork_items_h and args.fork_items_meta and args.fork_pairs):
            raise SystemExit(
                f"Method {method} requires --fork_items_h, --fork_items_meta, --fork_pairs."
            )
        items_h = np.load(args.fork_items_h).astype(np.float32)
        if items_h.shape[1] != hidden_dim:
            raise SystemExit(
                f"Fork items hidden_dim ({items_h.shape[1]}) != PRM800K ({hidden_dim})."
            )
        anchor_idx, pos_idx, neg_idx = load_fork_pairs(args.fork_items_meta, args.fork_pairs)
        if args.max_pairs is not None:
            anchor_idx = anchor_idx[: args.max_pairs]
            pos_idx = pos_idx[: args.max_pairs]
            neg_idx = neg_idx[: args.max_pairs]
        if method.endswith("rank"):
            objective = "rank"
        elif method.endswith("triplet"):
            objective = "triplet"
        else:  # ae_recon / sae_recon -> matched recon-only control
            objective = "none"
        # ae_* => dense (no sparsity); sae_* => keep L1 sparsity penalty.
        rep_l1 = 0.0 if method.startswith("ae") else args.l1_weight
        rep_train_n = int(pos_idx.shape[0])
        t0 = time.time()
        sae, sae_stats = train_repr_with_pairs(
            items_h=items_h,
            anchor_idx=anchor_idx, pos_idx=pos_idx, neg_idx=neg_idx,
            hidden_dim=hidden_dim, latent_dim=latent_dim,
            epochs=args.epochs_sae, batch_size=args.batch_size, lr=args.lr_sae,
            l1_weight=rep_l1, objective=objective, obj_weight=args.obj_weight,
            rank_kind=args.rank_kind, rank_margin=args.rank_margin,
            triplet_metric=args.triplet_metric, triplet_margin=args.triplet_margin,
            device=device, seed=args.seed,
        )
        representation_train_time = time.time() - t0
        torch.save(sae.state_dict(), args.out_dir / "representation.pt")
        for p in sae.parameters():
            p.requires_grad_(False)
        sae.eval()

    # ---- Build probe inputs for training/val (not ProcessBench yet) -------
    if sae is not None:
        z_probe_train = encode_with_sae(sae, probe_train_h, args.batch_size, device)
        z_val = encode_with_sae(sae, val_h, args.batch_size, device)
    else:
        z_probe_train = probe_train_h
        z_val = val_h

    # ---- Final probe -------------------------------------------------------
    if method == "random":
        rng = np.random.default_rng(args.seed)
        val_scores = rng.uniform(0.0, 1.0, size=val_h.shape[0]).astype(np.float32)
        probe = None
    else:
        t0 = time.time()
        probe = train_linear_probe(
            z_train=z_probe_train,
            y_train=probe_train_y,
            z_val=z_val,
            y_val=val_y,
            epochs=args.epochs_probe,
            batch_size=args.batch_size,
            lr=args.lr_probe,
            patience=args.early_stopping_patience,
            device=device,
            seed=args.seed,
        )
        probe_train_time = time.time() - t0
        torch.save(probe.state_dict(), args.out_dir / "linear_probe.pt")
        val_scores = probe_scores(probe, z_val, args.batch_size, device)

    # Random method: zero out timing per spec
    if method == "random":
        representation_train_time = 0.0
        probe_train_time = 0.0

    # ---- Threshold selection ----------------------------------------------
    np.save(args.out_dir / "val_scores.npy", val_scores)
    val_selected_threshold, best_bacc, val_f1 = select_threshold(
        val_scores, val_y, threshold_grid
    )
    (args.out_dir / "threshold.json").write_text(
        json.dumps(
            {
                "selected_threshold": val_selected_threshold,
                "selection_metric": "balanced_accuracy",
                "best_val_balanced_accuracy": best_bacc,
                "val_f1_binary": val_f1,
                "threshold_grid": threshold_grid,
                "fixed_threshold_0p5": 0.5,
            },
            indent=2,
        )
    )

    # ---- ProcessBench eval (loop over targets, emit per-name files) -------
    all_eval_summaries: list[dict] = []
    for pb_name, pb_h_arr, pb_meta_rows in pb_targets:
        if device.type == "cuda":
            torch.cuda.synchronize()
        eval_t0 = time.time()

        if sae is not None:
            z_pb = encode_with_sae(sae, pb_h_arr, args.batch_size, device)
        else:
            z_pb = pb_h_arr

        if method == "random":
            pb_rng = np.random.default_rng(args.seed + 1)
            pb_scores = pb_rng.uniform(0.0, 1.0, size=pb_h_arr.shape[0]).astype(np.float32)
        else:
            pb_scores = probe_scores(probe, z_pb, args.batch_size, device)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Oracle threshold = grid-search on PB itself (per-step labels: is this
        # step the first error?). For trace-level F1_PB the oracle is computed
        # below by scanning the same grid for the value that maximizes F1_PB.
        eval_time_inference = time.time() - eval_t0

        # First-error step labels per row, used for diagnostic stepwise selection.
        # row['label'] is trace-level first-error idx; per-step "is-error" label is
        # 1 iff row['step_idx'] == row['label'].
        step_is_error = np.array(
            [int(int(r["step_idx"]) == int(r["label"])) for r in pb_meta_rows],
            dtype=np.int64,
        )

        def eval_at(t: float) -> tuple[list[dict], dict, float]:
            t0 = time.time()
            rows, m = evaluate_processbench(pb_scores, pb_meta_rows, t)
            return rows, m, time.time() - t0

        # 1. fixed t=0.5
        rows_fixed, m_fixed, dt_fixed = eval_at(0.5)
        # 2. val-selected
        rows_val, m_val, dt_val = eval_at(val_selected_threshold)
        # 3. oracle on PB (grid-search by F1_PB)
        best_oracle_t = threshold_grid[0]
        best_oracle_f1 = -1.0
        for t in threshold_grid:
            _, m_try = evaluate_processbench(pb_scores, pb_meta_rows, t)
            if m_try["F1_PB"] > best_oracle_f1:
                best_oracle_f1 = m_try["F1_PB"]
                best_oracle_t = t
        rows_oracle, m_oracle, dt_oracle = eval_at(best_oracle_t)

        # Persist scoring artifacts (use val-selected as the canonical scores file)
        with (args.out_dir / f"pb_step_scores_{pb_name}.jsonl").open("w") as f:
            for row in rows_val:
                f.write(json.dumps(row) + "\n")
        with (args.out_dir / f"pb_predictions_{pb_name}.jsonl").open("w") as f:
            for row in rows_val:
                f.write(json.dumps({
                    "id": row["id"], "label": row["label"],
                    "prediction": row["prediction"], "threshold": row["threshold"],
                }) + "\n")

        n_steps_total = pb_h_arr.shape[0]

        def make_metrics(tag: str, t: float, m: dict, dt: float, src: str) -> dict:
            eval_time = eval_time_inference + dt
            return {
                "method": method,
                "pb_name": pb_name,
                "threshold_type": tag,
                "threshold": t,
                "threshold_selection_source": src,
                **m,
                "eval_time_sec": eval_time,
                "mean_step_latency_ms": eval_time * 1000.0 / max(n_steps_total, 1),
                "mean_trace_latency_ms": eval_time * 1000.0 / max(m["n_traces"], 1),
            }

        em_fixed = make_metrics("fixed_t0.5", 0.5, m_fixed, dt_fixed, "fixed")
        em_val = make_metrics("val_selected", val_selected_threshold, m_val, dt_val, "val_balanced_accuracy")
        em_oracle = make_metrics("oracle", best_oracle_t, m_oracle, dt_oracle,
                                 "oracle_grid_max_F1_PB_on_processbench (NOT DEPLOYABLE)")

        for em, suffix in [
            (em_fixed, "fixed_t0.5"),
            (em_val, "val_selected"),
            (em_oracle, "oracle"),
        ]:
            (args.out_dir / f"eval_metrics_{pb_name}_{suffix}.json").write_text(
                json.dumps(em, indent=2)
            )
            all_eval_summaries.append(em)

        print(
            f"[{method}|{pb_name}] "
            f"t0.5 F1={em_fixed['F1_PB']:.4f}  "
            f"val(t={val_selected_threshold}) F1={em_val['F1_PB']:.4f}  "
            f"oracle(t={best_oracle_t}) F1={em_oracle['F1_PB']:.4f}"
        )

    # Aggregate eval summary (for downstream leaderboard merge)
    (args.out_dir / "eval_summary.json").write_text(
        json.dumps({"runs": all_eval_summaries}, indent=2)
    )
    # Preserve a representative legacy file for callers that still look for it:
    if all_eval_summaries:
        primary = next(
            (e for e in all_eval_summaries if e["threshold_type"] == "val_selected"),
            all_eval_summaries[0],
        )
        (args.out_dir / "eval_metrics.json").write_text(json.dumps(primary, indent=2))

    # ---- Train metrics ----------------------------------------------------
    total_time = time.time() - total_t0
    if method == "random":
        total_time = 0.0

    train_metrics = {
        "method": method,
        "seed": args.seed,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "representation_train_n": rep_train_n,
        "probe_train_n": int(probe_train_h.shape[0]),
        "val_n": int(val_h.shape[0]),
        "representation_train_time_sec": representation_train_time,
        "probe_train_time_sec": probe_train_time,
        "total_train_time_sec": total_time,
        "avg_representation_train_latency_ms_per_example": (
            (representation_train_time * 1000.0 / rep_train_n) if rep_train_n else 0.0
        ),
        "avg_probe_train_latency_ms_per_example": (
            (probe_train_time * 1000.0 / probe_train_h.shape[0])
            if method != "random"
            else 0.0
        ),
        "final_reconstruction_mse": (
            sae_stats["final_reconstruction_mse"] if method != "dense_linear" and method != "random" else None
        ),
        "final_l1_mean": (
            sae_stats["final_l1_mean"] if method != "dense_linear" and method != "random" else None
        ),
        "final_aux_bce": sae_stats["final_aux_bce"],
        # Sprint 2 objective diagnostics (None for non-fork methods).
        "objective": sae_stats.get("objective"),
        "obj_weight": sae_stats.get("obj_weight"),
        "final_objective_loss": sae_stats.get("final_objective_loss"),
        "final_pair_accuracy": sae_stats.get("final_pair_accuracy"),
        "final_margin_satisfaction": sae_stats.get("final_margin_satisfaction"),
        "gpu_name": gpu_name,
        "device": device.type,
    }
    (args.out_dir / "train_metrics.json").write_text(json.dumps(train_metrics, indent=2))

    # Per-epoch representation-training curves (recon / l1 / objective /
    # pair_accuracy / margin_satisfaction) for the fork methods.
    if sae_stats.get("history"):
        (args.out_dir / "representation_history.json").write_text(
            json.dumps(sae_stats["history"], indent=2)
        )

    if all_eval_summaries:
        primary = next(
            (e for e in all_eval_summaries if e["threshold_type"] == "val_selected"),
            all_eval_summaries[0],
        )
        print(
            f"[{method}|{primary['pb_name']}|val_selected] "
            f"threshold={primary['threshold']} F1_PB={primary['F1_PB']:.4f} "
            f"Acc_err={primary['Acc_error']:.4f} Acc_corr={primary['Acc_correct']:.4f} "
            f"val_bacc={best_bacc:.4f}"
        )


if __name__ == "__main__":
    main()
