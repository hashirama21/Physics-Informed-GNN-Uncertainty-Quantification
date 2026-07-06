"""
KG-GNN — Training and evaluation.

Objective (Eq. totalloss, extended):
    L = L_CE + l_phys * L_phys + l_obs * L_obs + l_T * L_T + l_ea * Omega(Ea)

L_obs and L_T counteract the trivial optimum T -> T_max of L_phys alone:
the encoder must infer the lowest hot-spot temperature consistent with the
observed gas signature.

Evaluation protocol: the test data is touched exactly once per run;
calibration temperature is fitted on validation only. `--cv-folds/--cv-repeats`
switches to repeated stratified k-fold (paper protocol).

Ablations: --no-gates --no-phys --no-obs --free-ea --frozen-ea

Usage:
  python train_kg.py                              # single split
  python train_kg.py --cv-folds 5 --cv-repeats 5  # paper protocol
  python train_kg.py --no-obs --cv-folds 5 --cv-repeats 5

Author : DONGMO
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

from models.kg_gnn import KGGNN, build_kg_model
from models.kinetics import EDGE_IDX, REACTION_CHANNELS
from models.preprocessing import (
    build_graph_dataset, compute_duval_coords, extract_and_validate_ratios,
    handle_zeros_and_missing, load_dataset, run_preprocessing,
)
from utils.config import (
    CLASS_WEIGHTS, DATA_PATH, DEVICE, IDX_TO_LABEL, NUM_CLASSES,
    OUTPUT_DIR, TARGET_COL, TRAIN_CONFIG, get_logger, set_seed,
)

logger = get_logger("train_kg")

LABEL_SMOOTHING = 0.02
DISCHARGE_CLASSES = [1, 5]      # D2, DT
HIGH_ENERGY_CLASSES = [1, 4, 5]  # D2, T3, DT — classes whose regime opens the C2H2 channel
ACETYLENE_EDGE = EDGE_IDX["c2h4_c2h2"]

CV_SCALAR_KEYS = [
    "accuracy", "f1_macro", "precision", "recall", "min_class_recall",
    "kappa", "brier", "ece", "pcr", "gate_auroc", "gate_auroc_he",
    "mean_pred_entropy", "mean_mutual_info",
]


def graphs_to_tensors(graphs: list) -> TensorDataset:
    x = torch.stack([g.x for g in graphs])
    y = torch.cat([g.y for g in graphs])
    return TensorDataset(x, y)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    onehot = np.eye(y_prob.shape[1])[y_true]
    return float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    conf, preds = y_prob.max(axis=1), y_prob.argmax(axis=1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            ece += m.mean() * abs(conf[m].mean() - (preds[m] == y_true[m]).mean())
    return float(ece)


def classification_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_prob: np.ndarray) -> Dict:
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0,
                                    labels=list(range(NUM_CLASSES)))
    present = [r for c, r in enumerate(per_class_recall) if (y_true == c).any()]
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1_macro":  round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "min_class_recall": round(float(min(present)), 4),
        "kappa":     round(cohen_kappa_score(y_true, y_pred), 4),
        "brier":     round(brier_score(y_true, y_prob), 4),
        "ece":       round(ece_score(y_true, y_prob), 4),
        "per_class_recall": {IDX_TO_LABEL[i]: round(float(r), 4)
                             for i, r in enumerate(per_class_recall)},
    }


def train_one_epoch(model:     KGGNN,
                    loader:    DataLoader,
                    optimizer: torch.optim.Optimizer,
                    lambdas:   Dict[str, float],
                    mixup_alpha: float,
                    noise_std:   float,
                    ) -> Dict[str, float]:
    model.train()
    cw = CLASS_WEIGHTS.to(DEVICE)
    tot = {"loss": 0.0, "ce": 0.0, "phys": 0.0, "obs": 0.0, "thinge": 0.0, "ea": 0.0}
    n_batches = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        x_clean = x
        if noise_std > 0:
            x = x + noise_std * torch.randn_like(x)

        optimizer.zero_grad()
        out = model(x)
        logits_clean = out["logits"]

        phys = model.consistency_loss(F.softmax(logits_clean, dim=-1), out["kappa"])
        # absence scores from the un-noised measurements
        obs = model.observation_loss(x_clean, out["kappa"])
        thinge = model.temperature_hinge(out["T"], y)
        ea_pen = model.gates.elastic_penalty()

        if mixup_alpha > 0 and x.size(0) > 1:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx = torch.randperm(x.size(0), device=DEVICE)
            logits_mix = model.head(lam * out["embed"] + (1 - lam) * out["embed"][idx])
            ce = (lam * F.cross_entropy(logits_mix, y, weight=cw,
                                        label_smoothing=LABEL_SMOOTHING)
                  + (1 - lam) * F.cross_entropy(logits_mix, y[idx], weight=cw,
                                                label_smoothing=LABEL_SMOOTHING))
        else:
            ce = F.cross_entropy(logits_clean, y, weight=cw,
                                 label_smoothing=LABEL_SMOOTHING)

        loss = (ce
                + lambdas["phys"] * phys
                + lambdas["obs"] * obs
                + lambdas["thinge"] * thinge
                + lambdas["ea"] * ea_pen)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["grad_clip"])
        optimizer.step()

        for k, v in zip(["loss", "ce", "phys", "obs", "thinge", "ea"],
                        [loss, ce, phys, obs, thinge, ea_pen]):
            tot[k] += v.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in tot.items()}


@torch.no_grad()
def evaluate(model: KGGNN, loader: DataLoader) -> Dict:
    model.eval()
    ys, preds, probs = [], [], []
    for x, y in loader:
        p = F.softmax(model(x.to(DEVICE))["logits"], dim=-1).cpu().numpy()
        probs.append(p)
        preds.append(p.argmax(axis=1))
        ys.append(y.numpy())
    return classification_metrics(np.concatenate(ys), np.concatenate(preds),
                                  np.concatenate(probs))


@torch.no_grad()
def evaluate_mc(model:       KGGNN,
                loader:      DataLoader,
                n_samples:   int,
                temperature: float = 1.0,
                ) -> Dict:
    model.eval()
    ys, probs, ents, mis, pcr_sum = [], [], [], [], 0.0
    temps, kappa_c2h2 = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        mc = model.mc_dropout_predict(x, n_samples=n_samples)
        p = mc["mean_probs"]
        if temperature != 1.0:
            p = F.softmax(torch.log(p.clamp(min=1e-9)) / temperature, dim=-1)
        probs.append(p.cpu().numpy())
        ys.append(y.numpy())
        ents.append(mc["pred_entropy"].cpu().numpy())
        mis.append(mc["epistemic"].cpu().numpy())
        pcr_sum += float(model.pcr(p.argmax(dim=1), mc["kappa"])) * len(y)
        temps.append(mc["T"].squeeze(-1).cpu().numpy())
        kappa_c2h2.append(mc["kappa"][:, ACETYLENE_EDGE].cpu().numpy())

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(probs)
    m = classification_metrics(y_true, y_prob.argmax(axis=1), y_prob)
    m["pcr"] = round(pcr_sum / len(y_true), 4)
    m["mean_pred_entropy"] = round(float(np.concatenate(ents).mean()), 4)
    m["mean_mutual_info"] = round(float(np.concatenate(mis).mean()), 4)

    # Gate Discriminativeness: kappa(C2H4->C2H2) as binary detector.
    # `gate_auroc` targets discharge classes (D2/DT); `gate_auroc_he` targets the
    # high-energy regime (D2/T3/DT), which is what the acetylene gate physically
    # encodes — T3 legitimately opens it, so the first AUROC is bounded by design.
    k_arc = np.concatenate(kappa_c2h2)

    def _auroc(pos_classes):
        pos = np.isin(y_true, pos_classes)
        return (round(float(roc_auc_score(pos, k_arc)), 4)
                if pos.any() and (~pos).any() else float("nan"))

    m["gate_auroc"] = _auroc(DISCHARGE_CLASSES)
    m["gate_auroc_he"] = _auroc(HIGH_ENERGY_CLASSES)

    t_all = np.concatenate(temps)
    m["T_by_class"] = {
        IDX_TO_LABEL[c]: {"mean": round(float(t_all[y_true == c].mean()), 1),
                          "std":  round(float(t_all[y_true == c].std()), 1)}
        for c in range(NUM_CLASSES) if (y_true == c).any()
    }
    return m


def fit_temperature(model: KGGNN, loader: DataLoader) -> float:
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x.to(DEVICE))["logits"].cpu())
            targets.append(y)
    logits, targets = torch.cat(logits), torch.cat(targets)
    T = nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=0.05), targets)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.clamp(min=0.1, max=10.0).item())


def run_single(args,
               lambdas: Dict[str, float],
               train_g: list, val_g: list, test_g: list,
               ckpt_path,
               verbose: bool = True,
               ) -> Dict:
    train_loader = DataLoader(graphs_to_tensors(train_g),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(graphs_to_tensors(val_g), batch_size=args.batch_size)
    test_loader = DataLoader(graphs_to_tensors(test_g), batch_size=args.batch_size)

    model = build_kg_model(hidden_dim=args.hidden, num_layers=args.layers,
                           dropout=args.dropout, learn_ea=not args.frozen_ea,
                           use_gates=not args.no_gates).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=TRAIN_CONFIG["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=1e-6)

    best_f1, patience_left, epochs_run = -1.0, args.patience, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, lambdas,
                             args.mixup_alpha, args.noise_std)
        va = evaluate(model, val_loader)
        scheduler.step()
        epochs_run = epoch

        if verbose and (epoch == 1 or epoch % 25 == 0):
            logger.info(f"Ep {epoch:03d} | loss={tr['loss']:.4f} ce={tr['ce']:.4f} "
                        f"phys={tr['phys']:.4f} obs={tr['obs']:.4f} "
                        f"tT={tr['thinge']:.4f} ea={tr['ea']:.4f} | "
                        f"val F1={va['f1_macro']:.4f} | {time.time() - t0:.1f}s")

        if va["f1_macro"] > best_f1 + 1e-4:
            best_f1, patience_left = va["f1_macro"], args.patience
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    t_cal = fit_temperature(model, val_loader)

    # Validation diagnostics — the only numbers hyperparameters may be tuned on
    val_mc = evaluate_mc(model, val_loader, n_samples=args.mc_samples,
                         temperature=t_cal)
    test_det = evaluate(model, test_loader)
    test_mc = evaluate_mc(model, test_loader, n_samples=args.mc_samples,
                          temperature=t_cal)

    ea_learned = model.gates.ea.detach().cpu()
    ea_report = {
        c.name: {"learned_kJ_mol": round(float(ea_learned[i]), 2),
                 "interval": [c.ea_lo, c.ea_hi], "t_half_K": c.t_half}
        for i, c in enumerate(REACTION_CHANNELS)
    }
    return {
        "best_val_f1": round(best_f1, 4),
        "calibration_temperature": round(t_cal, 4),
        "epochs_run": epochs_run,
        "val_mc_dropout": val_mc,
        "test_deterministic": test_det,
        "test_mc_dropout": test_mc,
        "activation_energies": ea_report,
    }


def run_cv(args, lambdas: Dict[str, float], tag: str) -> Dict:
    df = load_dataset(DATA_PATH)
    df = handle_zeros_and_missing(df)
    df = extract_and_validate_ratios(df)

    rskf = RepeatedStratifiedKFold(n_splits=args.cv_folds,
                                   n_repeats=args.cv_repeats,
                                   random_state=args.seed)
    n_total = args.cv_folds * args.cv_repeats
    ckpt_path = OUTPUT_DIR / f"{tag}_cv_tmp.pt"
    fold_results: List[Dict] = []

    for fold_i, (tr_idx, te_idx) in enumerate(rskf.split(df, df[TARGET_COL]), 1):
        set_seed(args.seed + fold_i)
        df_trval = df.iloc[tr_idx].reset_index(drop=True)
        df_te = df.iloc[te_idx].reset_index(drop=True)
        try:
            df_tr, df_va = train_test_split(df_trval, test_size=0.15,
                                            stratify=df_trval[TARGET_COL],
                                            random_state=args.seed + fold_i)
        except ValueError:  # a minority class too small to stratify
            df_tr, df_va = train_test_split(df_trval, test_size=0.15,
                                            random_state=args.seed + fold_i)

        df_tr = compute_duval_coords(df_tr.reset_index(drop=True))
        df_va = compute_duval_coords(df_va.reset_index(drop=True))
        df_te = compute_duval_coords(df_te)

        tr_g, scaler = build_graph_dataset(df_tr, fit_scaler=True)
        va_g, _ = build_graph_dataset(df_va, scaler=scaler)
        te_g, _ = build_graph_dataset(df_te, scaler=scaler)

        res = run_single(args, lambdas, tr_g, va_g, te_g, ckpt_path, verbose=False)
        mc = res["test_mc_dropout"]
        fold_results.append(mc)
        t_spread = {k: v["mean"] for k, v in mc["T_by_class"].items()}
        logger.info(f"[{tag}] fold {fold_i:02d}/{n_total} | "
                    f"F1={mc['f1_macro']:.4f} PCR={mc['pcr']:.3f} "
                    f"AUROC={mc['gate_auroc']:.3f}/{mc['gate_auroc_he']:.3f} | "
                    f"T={t_spread}")

    summary = {}
    for key in CV_SCALAR_KEYS:
        vals = [r[key] for r in fold_results if not np.isnan(r.get(key, np.nan))]
        summary[key] = {"mean": round(float(np.mean(vals)), 4),
                        "std":  round(float(np.std(vals)), 4)}

    t_agg = {}
    for lbl in IDX_TO_LABEL.values():
        means = [r["T_by_class"][lbl]["mean"] for r in fold_results
                 if lbl in r["T_by_class"]]
        if means:
            t_agg[lbl] = {"mean": round(float(np.mean(means)), 1),
                          "std": round(float(np.std(means)), 1)}

    logger.info(f"== [{tag}] {args.cv_folds}x{args.cv_repeats} CV summary ==")
    for k, s in summary.items():
        logger.info(f"  {k:18s}: {s['mean']:.4f} +/- {s['std']:.4f}")
    logger.info(f"  T by class: {t_agg}")

    ckpt_path.unlink(missing_ok=True)
    return {"summary": summary, "T_by_class": t_agg, "folds": fold_results}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train KG-GNN on the DGA dataset")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=TRAIN_CONFIG["batch_size"])
    ap.add_argument("--lr", type=float, default=TRAIN_CONFIG["learning_rate"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--lambda-phys", type=float, default=0.1)
    ap.add_argument("--lambda-obs", type=float, default=0.2)
    ap.add_argument("--lambda-thinge", type=float, default=0.3)
    ap.add_argument("--lambda-ea", type=float, default=1.0)
    ap.add_argument("--mixup-alpha", type=float, default=TRAIN_CONFIG["mixup_alpha"])
    ap.add_argument("--noise-std", type=float, default=TRAIN_CONFIG["noise_std"])
    ap.add_argument("--mc-samples", type=int, default=50)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--seed", type=int, default=TRAIN_CONFIG["random_seed"])
    ap.add_argument("--cv-folds", type=int, default=0, help=">0 enables repeated k-fold")
    ap.add_argument("--cv-repeats", type=int, default=1)
    ap.add_argument("--no-gates", action="store_true")
    ap.add_argument("--no-phys", action="store_true")
    ap.add_argument("--no-obs", action="store_true")
    ap.add_argument("--free-ea", action="store_true")
    ap.add_argument("--frozen-ea", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    lambdas = {
        "phys":   0.0 if args.no_phys else args.lambda_phys,
        "obs":    0.0 if args.no_obs else args.lambda_obs,
        "thinge": 0.0 if args.no_obs else args.lambda_thinge,
        "ea":     0.0 if args.free_ea else args.lambda_ea,
    }
    tag = ("kg_gnn"
           + ("_nogates" if args.no_gates else "")
           + ("_nophys" if args.no_phys else "")
           + ("_noobs" if args.no_obs else "")
           + ("_freeea" if args.free_ea else "")
           + ("_frozenea" if args.frozen_ea else ""))

    logger.info("=" * 60)
    logger.info(f"KG-GNN — variant: {tag} | lambdas={lambdas} | "
                f"gates={'off' if args.no_gates else 'on'}")
    logger.info("=" * 60)

    if args.cv_folds > 0:
        report = {"variant": tag, "config": vars(args), "protocol":
                  f"{args.cv_folds}-fold x {args.cv_repeats} repeats",
                  **run_cv(args, lambdas, tag)}
        out_path = OUTPUT_DIR / f"{tag}_cv_report.json"
    else:
        train_g, val_g, test_g, *_ = run_preprocessing(DATA_PATH)
        res = run_single(args, lambdas, train_g, val_g, test_g,
                         OUTPUT_DIR / f"{tag}_best.pt")
        logger.info("== Test (MC Dropout, calibrated) ==")
        for k, v in res["test_mc_dropout"].items():
            logger.info(f"  {k}: {v}")
        report = {"variant": tag, "config": vars(args),
                  "dataset": {"train": len(train_g), "val": len(val_g),
                              "test": len(test_g)}, **res}
        out_path = OUTPUT_DIR / f"{tag}_report.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved: {out_path}")


if __name__ == "__main__":
    main()
