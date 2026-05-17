"""
PIGNN-UQ — Training, cross-validation, and evaluation
Reference: Chapter 3 — DONGMO

Author : DONGMO
GitHub : https://github.com/hashirama21
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score,
    f1_score, precision_score, recall_score,
)

from collections import Counter

import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
import pandas as pd
from utils.config import (
    CLASS_WEIGHTS, DATA_PATH, DEVICE, EVAL_TARGETS,
    FAULT_CODES, FAULT_LABELS, GAS_TO_VIT, IDX_TO_CODE, IDX_TO_LABEL,
    IEEE_THRESHOLDS, MODEL_CONFIG, NUM_CLASSES, OUTPUT_DIR, TRAIN_CONFIG,
    estimate_rul, get_logger, rul_decision, set_seed,
)
from models.models import PIGNN_UQ, build_model
from models.preprocessing import (
    build_graph_dataset, compute_duval_coords,
    get_kfold_splits, run_preprocessing,
)

logger = get_logger("train")


def train_one_epoch(model:     PIGNN_UQ,
                    loader:    DataLoader,
                    optimizer: torch.optim.Optimizer,
                    use_class_weights: bool = True,
                    grad_clip: float = TRAIN_CONFIG["grad_clip"]
                    ) -> Dict[str, float]:
    model.train()
    tot_loss = tot_ce = tot_phys = 0.0
    n = 0
    cw = CLASS_WEIGHTS if use_class_weights else None

    noise_std    = TRAIN_CONFIG.get("noise_std", 0.0)
    drop_edge_p  = TRAIN_CONFIG.get("drop_edge_p", 0.0)
    mixup_alpha  = TRAIN_CONFIG.get("mixup_alpha", 0.0)
    for batch in loader:
        batch = batch.to(DEVICE)
        if noise_std > 0:
            batch.x = batch.x + noise_std * torch.randn_like(batch.x)
        if drop_edge_p > 0:
            mask = torch.rand(batch.edge_index.shape[1], device=DEVICE) > drop_edge_p
            batch.edge_index = batch.edge_index[:, mask]
            if batch.edge_attr is not None:
                batch.edge_attr = batch.edge_attr[mask]
        optimizer.zero_grad()

        batch_size = int(batch.batch.max().item()) + 1 if batch.batch is not None else 1
        if mixup_alpha > 0 and batch_size > 1:
            emb = model.forward_embed(batch)               # [B, hidden_dim]
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx = torch.randperm(emb.size(0), device=DEVICE)
            mixed = lam * emb + (1.0 - lam) * emb[idx]
            logits = model.classifier(mixed)
            y_a = batch.y.to(DEVICE)
            y_b = batch.y[idx].to(DEVICE)
            cw_dev = cw.to(DEVICE) if cw is not None else None
            loss = (lam * F.cross_entropy(logits, y_a, weight=cw_dev)
                    + (1.0 - lam) * F.cross_entropy(logits, y_b, weight=cw_dev))
            tot_ce   += loss.item()
            tot_loss += loss.item()
        else:
            logits, _  = model(batch)
            loss_dict  = model.compute_loss(logits, batch.y.to(DEVICE), batch,
                                             class_weights=cw)
            loss = loss_dict["total"]
            tot_ce   += loss_dict["ce"].item()
            tot_phys += loss_dict["physics"].item()
            tot_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        n += 1

    d = max(n, 1)
    return {"loss": tot_loss / d, "ce": tot_ce / d, "physics": tot_phys / d}


def evaluate(model:  PIGNN_UQ,
             loader: DataLoader,
             use_mc: bool = False
             ) -> Dict[str, float]:
    model.eval()
    all_preds, all_targets, all_probs, all_uncerts = [], [], [], []

    for batch in loader:
        batch   = batch.to(DEVICE)
        targets = batch.y.cpu().numpy()

        if use_mc:
            with torch.no_grad():
                mc = model.mc_dropout_predict(batch, n_samples=MODEL_CONFIG["mc_samples"])
            preds   = mc["pred_class"].cpu().numpy()
            probs   = mc["mean_probs"].cpu().numpy()
            uncerts = mc["uncertainty"].cpu().numpy()
        else:
            with torch.no_grad():
                logits, _ = model(batch)
            probs_t = torch.softmax(logits, dim=-1).cpu().numpy()
            preds   = probs_t.argmax(axis=1)
            probs   = probs_t
            uncerts = np.zeros(len(preds))

        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())
        all_probs.extend(probs.tolist())
        all_uncerts.extend(uncerts.tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    m = _classification_metrics(y_true, y_pred, y_prob)
    m["mean_uncertainty"] = float(np.mean(all_uncerts))
    return m


def _classification_metrics(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred,    average="macro", zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred,        average="macro", zero_division=0), 4),
        "kappa":     round(cohen_kappa_score(y_true, y_pred), 4),
        "brier":     round(_brier_score(y_true, y_prob), 4),
        "ece":       round(_ece(y_true, y_prob), 4),
    }


def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    onehot = np.zeros_like(y_prob)
    for i, t in enumerate(y_true):
        onehot[i, int(t)] = 1.0
    return float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    conf  = y_prob.max(axis=1)
    preds = y_prob.argmax(axis=1)
    bins  = np.linspace(0, 1, n_bins + 1)
    ece   = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        ece += m.sum() / len(y_true) * abs(conf[m].mean() - (preds[m] == y_true[m]).mean())
    return float(ece)


def find_temperature(model: PIGNN_UQ, val_loader: DataLoader) -> float:
    """Optimise T pour minimiser la NLL sur val (temperature scaling post-hoc)."""
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            logits, _ = model(batch)
            all_logits.append(logits.cpu())
            all_targets.append(batch.y.cpu())
    logits  = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=200)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=0.01), targets)
        loss.backward()
        return loss
    opt.step(closure)
    t_val = float(T.clamp(min=0.1, max=10.0).item())
    logger.info(f"Temperature scaling: T={t_val:.4f}")
    return t_val


def find_ensemble_temperature(ens_log_probs: torch.Tensor,
                               targets:        torch.Tensor) -> float:
    """Optimise T for ensemble log-probabilities on val set."""
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=200)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(ens_log_probs / T.clamp(min=0.01), targets)
        loss.backward()
        return loss
    opt.step(closure)
    t_val = float(T.clamp(min=0.5, max=3.0).item())
    logger.info(f"Ensemble temperature scaling: T={t_val:.4f}")
    return t_val


def ensemble_evaluate(fold_paths:     List[str],
                      test_graphs:     list,
                      val_graphs:      list,
                      node_in_dim:     int,
                      fold_weights:    Optional[List[float]] = None,
                      ) -> Dict[str, float]:
    """Weighted average of MC Dropout probs from all fold models on test set.
    Temperature is optimised separately for the ensemble on val set."""
    all_model_probs: list    = []
    all_val_probs: list      = []
    all_targets: list        = []
    valid_weights: list      = []

    test_loader = DataLoader(test_graphs, batch_size=TRAIN_CONFIG["batch_size"])
    val_loader  = DataLoader(val_graphs,  batch_size=TRAIN_CONFIG["batch_size"])
    val_targets: list = []

    for i, path in enumerate(fold_paths):
        if not os.path.exists(path):
            logger.warning(f"Fold model not found: {path}")
            continue
        m = build_model(node_in_dim=node_in_dim).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))

        fold_probs, fold_tgts = [], []
        for batch in test_loader:
            batch = batch.to(DEVICE)
            mc = m.mc_dropout_predict(batch, n_samples=MODEL_CONFIG["mc_samples"])
            fold_probs.append(mc["mean_probs"].cpu())
            fold_tgts.extend(batch.y.cpu().tolist())
        all_model_probs.append(torch.cat(fold_probs, dim=0))
        if not all_targets:
            all_targets = fold_tgts

        # Collect val predictions for ensemble T-scaling
        vp = []
        for batch in val_loader:
            batch = batch.to(DEVICE)
            mc = m.mc_dropout_predict(batch, n_samples=MODEL_CONFIG["mc_samples"])
            vp.append(mc["mean_probs"].cpu())
            if i == 0:
                val_targets.extend(batch.y.cpu().tolist())
        all_val_probs.append(torch.cat(vp, dim=0))

        w = fold_weights[i] if (fold_weights and i < len(fold_weights)) else 1.0
        valid_weights.append(max(w, 1e-9))

    if not all_model_probs:
        logger.error("No fold models found — skipping ensemble.")
        return {}

    # Weighted average of probabilities
    weights_t = torch.tensor(valid_weights, dtype=torch.float)
    weights_t = weights_t / weights_t.sum()
    stacked   = torch.stack(all_model_probs, dim=0)          # [K, N, C]
    ens_probs = (stacked * weights_t[:, None, None]).sum(dim=0)

    # Ensemble-specific temperature scaling on val set
    val_stacked  = torch.stack(all_val_probs, dim=0)
    val_ens_prob = (val_stacked * weights_t[:, None, None]).sum(dim=0)
    val_log_p    = torch.log(val_ens_prob.clamp(min=1e-9))
    val_tgts_t   = torch.tensor(val_targets, dtype=torch.long)
    ens_T        = find_ensemble_temperature(val_log_p, val_tgts_t)

    ens_log = torch.log(ens_probs.clamp(min=1e-9))
    ens_probs_cal = torch.softmax(ens_log / ens_T, dim=-1)

    y_prob = ens_probs_cal.numpy()
    y_pred = y_prob.argmax(axis=1)
    y_true = np.array(all_targets)

    m_dict = _classification_metrics(y_true, y_pred, y_prob)
    m_dict["mean_uncertainty"] = 0.0
    return m_dict


def check_targets(metrics: Dict[str, float]) -> bool:
    ok = all(metrics.get(k, 0) >= v
             for k, v in EVAL_TARGETS.items() if k not in ("brier", "ece"))
    return (ok
            and metrics.get("brier", 1.0) <= EVAL_TARGETS["brier"]
            and metrics.get("ece",   1.0) <= EVAL_TARGETS["ece"])


class EarlyStopping:
    def __init__(self, patience: int = TRAIN_CONFIG["early_stop_patience"],
                 delta: float = 1e-4, ckpt: str = "best.pt"):
        self.patience = patience
        self.delta    = delta
        self.ckpt     = ckpt
        self.best     = -np.inf
        self.counter  = 0
        self.stop     = False

    def __call__(self, score: float, model: PIGNN_UQ) -> bool:
        if score > self.best + self.delta:
            self.best    = score
            self.counter = 0
            torch.save(model.state_dict(), self.ckpt)
            logger.info(f"  Checkpoint saved (F1={score:.4f})")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                logger.info(f"  Early stopping triggered (patience={self.patience})")
        return self.stop


def train_split(train_graphs: list,
                val_graphs:   list,
                node_in_dim:  int = MODEL_CONFIG["node_in_dim"],
                fold_id:      int = 0
                ) -> Tuple[PIGNN_UQ, Dict]:
    set_seed(TRAIN_CONFIG["random_seed"])

    train_loader = DataLoader(train_graphs, batch_size=TRAIN_CONFIG["batch_size"],
                               shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_graphs,   batch_size=TRAIN_CONFIG["batch_size"],
                               shuffle=False)

    model     = build_model(node_in_dim=node_in_dim).to(DEVICE)
    optimizer = AdamW(model.parameters(),
                       lr=TRAIN_CONFIG["learning_rate"],
                       weight_decay=TRAIN_CONFIG["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=1, eta_min=1e-6)
    ckpt_path = os.path.join(OUTPUT_DIR, f"best_fold{fold_id}.pt")
    stopper   = EarlyStopping(ckpt=ckpt_path)
    history   = {"train_loss": [], "val_f1": [], "val_acc": [], "lr": []}

    logger.info(f"== Fold {fold_id} — training start ==")
    for epoch in range(1, TRAIN_CONFIG["num_epochs"] + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer)
        va = evaluate(model, val_loader, use_mc=False)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr["loss"])
        history["val_f1"].append(va["f1"])
        history["val_acc"].append(va["accuracy"])
        history["lr"].append(lr)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Ep {epoch:03d}/{TRAIN_CONFIG['num_epochs']} | "
                f"Loss={tr['loss']:.4f} CE={tr['ce']:.4f} Phys={tr['physics']:.4f} | "
                f"Val F1={va['f1']:.4f} Acc={va['accuracy']:.4f} | "
                f"LR={lr:.2e} | {time.time() - t0:.1f}s"
            )

        if stopper(va["f1"], model):
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    return model, history


def cross_validate(df_trainval,
                   node_in_dim: int = MODEL_CONFIG["node_in_dim"]
                   ) -> Dict:
    splits       = get_kfold_splits(df_trainval)
    fold_metrics = []

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"FOLD {fold_id + 1}/{len(splits)}  "
                    f"train={len(tr_idx)}  val={len(va_idx)}")

        df_tr = df_trainval.iloc[tr_idx].reset_index(drop=True)
        df_va = df_trainval.iloc[va_idx].reset_index(drop=True)

        # Duval coordinates computed per fold to prevent data leakage
        df_tr = compute_duval_coords(df_tr)
        df_va = compute_duval_coords(df_va)

        tr_g, scaler = build_graph_dataset(df_tr, fit_scaler=True)
        va_g, _      = build_graph_dataset(df_va, scaler=scaler)

        model, _ = train_split(tr_g, va_g, node_in_dim=node_in_dim, fold_id=fold_id)

        va_loader = DataLoader(va_g, batch_size=TRAIN_CONFIG["batch_size"])
        metrics   = evaluate(model, va_loader, use_mc=False)
        metrics["fold"] = fold_id + 1

        logger.info(f"  Fold {fold_id + 1} results:")
        for k, v in metrics.items():
            if k != "fold":
                logger.info(f"    {k:18s} = {v:.4f}")
        fold_metrics.append(metrics)

    summary = {}
    for key in ["accuracy", "f1", "precision", "recall", "kappa", "brier", "ece"]:
        vals = [m[key] for m in fold_metrics]
        summary[key] = {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals)),  4),
            "min":  round(float(np.min(vals)),  4),
            "max":  round(float(np.max(vals)),  4),
        }

    logger.info("\n== Cross-validation summary ==")
    for k, s in summary.items():
        logger.info(f"  {k:12s}: {s['mean']:.4f} ± {s['std']:.4f}")

    ok = check_targets({k: v["mean"] for k, v in summary.items()})
    logger.info(f"\n  Targets (Table 3.23): {'MET' if ok else 'NOT MET'}")
    return {"fold_metrics": fold_metrics, "summary": summary}


def inference_with_rul(model:   PIGNN_UQ,
                       graphs:  list,
                       df_test          # aligned DataFrame from run_preprocessing
                       ) -> List[Dict]:
    """
    MC Dropout inference with per-gas RUL estimation (IEEE C57.104 thresholds).
    Uses batched MC Dropout (~10x faster than sequential at batch_size=1).
    """
    model.eval()
    results = []

    for i, (g, (_, row)) in enumerate(zip(graphs, df_test.iterrows())):
        loader = DataLoader([g], batch_size=1)
        batch  = next(iter(loader)).to(DEVICE)

        mc = model.mc_dropout_predict_batched(batch, n_samples=MODEL_CONFIG["mc_samples"])

        pred_idx    = mc["pred_class"].item()
        pred_code   = IDX_TO_CODE[pred_idx]
        pred_label  = FAULT_LABELS[pred_code]
        uncertainty = mc["uncertainty"].item()
        confidence  = mc["confidence"].item()

        probs_dict = {
            FAULT_LABELS[FAULT_CODES[j]]: round(float(mc["mean_probs"][0, j]), 4)
            for j in range(NUM_CLASSES)
        }

        rul_estimates = {}
        for gas, thresh in IEEE_THRESHOLDS.items():
            if gas == "TDCG":
                current = float(row.get("TDCG", 0))
                vit_val = 0.0
            else:
                current = float(row.get(gas, 0))
                vit_col = GAS_TO_VIT.get(gas)
                vit_val = float(row[vit_col]) if vit_col and vit_col in row.index else 0.0

            rul = estimate_rul(current, vit_val, thresh, uncertainty)
            rul["decision"] = rul_decision(rul["rul_months"], uncertainty)
            rul_estimates[gas] = rul

        health = {}
        if hasattr(batch, "health_indices") and batch.health_indices is not None:
            hi     = batch.health_indices[0].cpu().tolist()
            health = {"CDI": round(hi[0], 4), "TAI": round(hi[1], 4),
                      "DSI": round(hi[2], 4), "OHI": round(hi[3], 4)}

        true_code  = int(row.get("Classe", 0))
        true_label = FAULT_LABELS.get(true_code, "?")

        results.append({
            "sample_id":      i,
            "equipment":      str(row.get("equipement", f"sample_{i}")),
            "true_class":     true_label,
            "pred_class":     pred_label,
            "correct":        true_label == pred_label,
            "probabilities":  probs_dict,
            "uncertainty":    round(uncertainty, 4),
            "confidence":     round(confidence, 4),
            "rul_estimates":  rul_estimates,
            "health_indices": health,
        })

    acc = sum(r["correct"] for r in results) / max(len(results), 1)
    logger.info(f"Inference complete — {len(results)} samples — Acc={acc:.4f}")
    return results


def main():
    set_seed(TRAIN_CONFIG["random_seed"])
    logger.info("===============================================")
    logger.info("  PIGNN-UQ — Full pipeline")
    logger.info("===============================================")

    (train_graphs, val_graphs, test_graphs,
     scaler, df_train, df_val, df_test) = run_preprocessing(DATA_PATH)

    logger.info(f"Device: {DEVICE}")

    df_trainval = pd.concat([df_train, df_val], ignore_index=True)
    cv_results  = cross_validate(df_trainval, node_in_dim=MODEL_CONFIG["node_in_dim"])

    # Final model trained on 90% of train+val; 10% pseudo-val for early stopping & T-scaling
    import random as _rng
    _rng.seed(TRAIN_CONFIG["random_seed"])
    all_tv = train_graphs + val_graphs       # 284 graphs (already scaled)
    n_pv   = max(10, int(len(all_tv) * 0.10))   # ~28 pseudo-val samples
    _idx   = list(range(len(all_tv)))
    _rng.shuffle(_idx)
    final_train_g = [all_tv[i] for i in _idx[n_pv:]]   # ~256 samples
    final_val_g   = [all_tv[i] for i in _idx[:n_pv]]    # ~28 samples

    logger.info(f"\n== Final training on train+val 90% ({len(final_train_g)} samples) ==")
    final_model, history = train_split(
        train_graphs=final_train_g,
        val_graphs=final_val_g,
        node_in_dim=MODEL_CONFIG["node_in_dim"],
        fold_id=99,
    )
    torch.save(final_model.state_dict(), os.path.join(OUTPUT_DIR, "final_model.pt"))

    # Temperature scaling calibration sur pseudo-val
    val_loader_cal = DataLoader(final_val_g, batch_size=TRAIN_CONFIG["batch_size"])
    temperature    = find_temperature(final_model, val_loader_cal)

    test_loader  = DataLoader(test_graphs, batch_size=TRAIN_CONFIG["batch_size"])
    test_metrics = evaluate(final_model, test_loader, use_mc=True)

    logger.info("\n== Test set metrics — final model (MC Dropout) ==")
    for k, v in test_metrics.items():
        tgt = EVAL_TARGETS.get(k)
        tag = ""
        if tgt is not None:
            tag = " ✓" if ((k in ("brier", "ece") and v <= tgt)
                           or (k not in ("brier", "ece") and v >= tgt)) else " ✗"
        logger.info(f"  {k:20s} = {v:.4f}{tag}")

    # KFold ensemble sur test (weighted by fold val F1, with ensemble-specific T-scaling)
    fold_paths = [os.path.join(OUTPUT_DIR, f"best_fold{i}.pt")
                  for i in range(TRAIN_CONFIG["n_folds"])]
    fold_f1_weights = [m["f1"] for m in cv_results["fold_metrics"]]
    ens_metrics = ensemble_evaluate(fold_paths, test_graphs, val_graphs,
                                     MODEL_CONFIG["node_in_dim"],
                                     fold_weights=fold_f1_weights)
    logger.info("\n== Test set metrics — KFold Ensemble (weighted, T-scaled separately) ==")
    for k, v in ens_metrics.items():
        tgt = EVAL_TARGETS.get(k)
        tag = ""
        if tgt is not None:
            tag = " ✓" if ((k in ("brier", "ece") and v <= tgt)
                           or (k not in ("brier", "ece") and v >= tgt)) else " ✗"
        logger.info(f"  {k:20s} = {v:.4f}{tag}")

    rul_reports = inference_with_rul(final_model, test_graphs, df_test)

    report = {
        "dataset":            {"total": len(df_train) + len(df_val) + len(df_test),
                                "train": len(df_train), "val": len(df_val),
                                "test":  len(df_test)},
        "cross_validation":   cv_results["summary"],
        "test_metrics":       test_metrics,
        "ensemble_metrics":   ens_metrics,
        "temperature":        temperature,
        "targets_met":        check_targets(ens_metrics),
        "rul_sample":         rul_reports[:5],
    }
    out_path = os.path.join(OUTPUT_DIR, "pignn_uq_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\nJSON report: {out_path}")

    return final_model, test_metrics, rul_reports


if __name__ == "__main__":
    main()
