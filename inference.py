#!/usr/bin/env python3
"""
PIGNN-UQ — Standalone inference script
Physics-Informed Graph Attention Network with MC Dropout UQ

Input  : DGA gas concentrations (ppm) — single sample or CSV batch
Output : Fault class, probabilities, uncertainty, RUL, health indices

Author : Vincess Dongmo
GitHub : https://github.com/hashirama21/Physics-Informed-GNN-Uncertainty-Quantification

Usage:
  # Single sample (concentrations in ppm)
  python inference.py --h2 450 --ch4 120 --c2h2 8 --c2h4 65 --c2h6 45 --co 850 --co2 3200

  # With gas degradation rates (ppm/month) for RUL estimation
  python inference.py --h2 450 --ch4 120 --c2h2 8 --c2h4 65 --c2h6 45 --co 850 --co2 3200 \\
    --vit-h2 15 --vit-ch4 3 --vit-c2h2 0.8 --vit-c2h4 4 --vit-c2h6 2 --vit-co 25 --vit-co2 120

  # CSV batch mode
  python inference.py --csv samples.csv --output results.json

  # Use a specific fold model instead of the final model
  python inference.py --h2 450 ... --model outputs/best_fold3.pt

  # Use ensemble of all fold models (slowest, most accurate)
  python inference.py --h2 450 ... --ensemble
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.config import (
    DEVICE, FAULT_CODES, FAULT_LABELS, GAS_COLS, GAS_TO_VIT,
    IDX_TO_CODE, IEEE_THRESHOLDS, MODEL_CONFIG, NUM_CLASSES,
    OUTPUT_DIR, RATIO_COLS, RATIO_FORMULA, VIT_COLS,
    compute_cdi, compute_dsi, compute_ohi, compute_tai,
    estimate_rul, get_logger, rul_decision,
)
from models.models import build_model
from models.preprocessing import (
    DGAScaler, build_graph, compute_duval_coords,
    extract_and_validate_ratios, handle_zeros_and_missing, run_preprocessing,
)

logger = get_logger("inference")

SCALER_PATH    = OUTPUT_DIR / "scaler.pkl"
FINAL_MODEL_PT = OUTPUT_DIR / "final_model.pt"
FOLD_PATHS     = [OUTPUT_DIR / f"best_fold{i}.pt" for i in range(10)]

#  Fault class descriptions (IEEE / IEC)
FAULT_DESC = {
    "D1":  "Partial discharge (low energy)",
    "D2":  "Discharge of high energy",
    "T1":  "Thermal fault — low temperature (< 300 °C)",
    "T2":  "Thermal fault — medium temperature (300–700 °C)",
    "T3":  "Thermal fault — high temperature (> 700 °C)",
    "DT":  "Combined electrical + thermal fault",
}


# Scaler loading (fits on training data if not cached)

def _load_or_fit_scaler() -> DGAScaler:
    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        logger.info(f"Scaler loaded from {SCALER_PATH}")
        return scaler

    from utils.config import DATA_PATH
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No fitted scaler found at {SCALER_PATH} and no data file at {DATA_PATH}.\n"
            "Run 'python train.py' first to fit and save the scaler."
        )
    logger.info("Fitting scaler on training data (first run) …")
    from models.preprocessing import split_dataset, load_dataset
    df  = load_dataset(DATA_PATH)
    df  = handle_zeros_and_missing(df)
    df  = extract_and_validate_ratios(df)
    df_train, _, _ = split_dataset(df)
    df_train = compute_duval_coords(df_train)
    scaler   = DGAScaler().fit(df_train)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler fitted and saved to {SCALER_PATH}")
    return scaler


# Build pandas row from raw gas values

def _build_row(gases: Dict[str, float],
               vits:  Dict[str, float],
               equip: str = "unknown") -> pd.Series:
    """Assemble a single-row DataFrame compatible with preprocessing pipeline."""
    eps  = 1e-9
    row  = {}

    # Gas concentrations (ppm) — clamp negatives to detection limit
    for gas in GAS_COLS:
        row[gas] = max(float(gases.get(gas, 0.001)), 0.001)

    # Degradation rates (ppm/month)
    for gas in GAS_COLS:
        vit_col      = GAS_TO_VIT[gas]
        row[vit_col] = max(float(vits.get(gas, 0.0)), 0.0)

    # Compute ratios from raw gases
    for col, (num, den) in RATIO_FORMULA.items():
        row[col] = row[num] / (row[den] + eps)

    # TDCG
    row["TDCG"] = sum(row[g] for g in ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO"])

    # Cumulative columns (set to 0 — not available at inference time)
    for col in ["TDCGcumul", "TDCGdelta", "TDCGdeltanorm"]:
        row[col] = 0.0

    # Duval triangle coordinates
    tri = row["C2H2"] + row["C2H4"] + row["CH4"] + eps
    row["TauxC2H2"] = row["C2H2"] / tri
    row["TauxC2H4"] = row["C2H4"] / tri
    row["TauxCH4"]  = row["CH4"]  / tri

    # Dummy class label (required by build_graph but ignored during inference)
    row["Classe"]     = 201
    row["equipement"] = equip

    return pd.Series(row)


# Single-sample inference

def predict_single(model,
                   scaler:    DGAScaler,
                   gases:     Dict[str, float],
                   vits:      Dict[str, float],
                   equip:     str = "unknown",
                   n_mc:      int = MODEL_CONFIG["mc_samples"],
                   temperature: float = 1.0,
                   ) -> Dict:
    row   = _build_row(gases, vits, equip)
    graph = build_graph(row, scaler)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)

    loader = DataLoader([graph], batch_size=1)
    batch  = next(iter(loader)).to(DEVICE)

    mc = model.mc_dropout_predict(batch, n_samples=n_mc)

    pred_idx   = mc["pred_class"].item()
    pred_code  = IDX_TO_CODE[pred_idx]
    pred_label = FAULT_LABELS[pred_code]
    uncertainty = mc["uncertainty"].item()
    confidence  = mc["confidence"].item()

    # Temperature scaling
    probs_raw = mc["mean_probs"][0]
    log_p     = torch.log(probs_raw.clamp(min=1e-9))
    probs_cal = torch.softmax(log_p / max(temperature, 0.01), dim=-1)
    pred_idx  = probs_cal.argmax().item()
    pred_code = IDX_TO_CODE[pred_idx]
    pred_label = FAULT_LABELS[pred_code]

    probs_dict = {
        FAULT_LABELS[FAULT_CODES[j]]: round(float(probs_cal[j]), 4)
        for j in range(NUM_CLASSES)
    }

    # RUL per gas (IEEE C57.104)
    rul_estimates = {}
    for gas, thresh in IEEE_THRESHOLDS.items():
        if gas == "TDCG":
            current = float(row["TDCG"])
            vit_val = 0.0
        else:
            current = float(row[gas])
            vit_val = float(vits.get(gas, 0.0))
        rul_est = estimate_rul(current, vit_val, thresh, uncertainty)
        rul_est["decision"] = rul_decision(rul_est["rul_months"], uncertainty)
        rul_estimates[gas]  = rul_est

    # Health indices
    cdi = compute_cdi(float(row["CO"]),   float(row["CO2"]))
    tai = compute_tai(float(row["C2H4"]), float(row["C2H6"]))
    dsi = compute_dsi(float(row["H2"]),   float(row["C2H2"]))
    ohi = compute_ohi(float(row["TDCG"]), cdi, tai, dsi)

    return {
        "equipment":      equip,
        "pred_class":     pred_label,
        "pred_code":      pred_code,
        "description":    FAULT_DESC.get(pred_label, ""),
        "confidence":     round(confidence, 4),
        "uncertainty":    round(uncertainty, 4),
        "probabilities":  probs_dict,
        "health_indices": {
            "OHI": round(ohi, 4),
            "CDI": round(cdi, 4),
            "TAI": round(tai, 4),
            "DSI": round(dsi, 4),
        },
        "rul_estimates":  rul_estimates,
        "input_gases_ppm": {g: round(float(row[g]), 3) for g in GAS_COLS},
    }


# Ensemble inference (10 fold models × 50 MC passes)

def predict_ensemble(scaler:      DGAScaler,
                     gases:       Dict[str, float],
                     vits:        Dict[str, float],
                     equip:       str = "unknown",
                     n_mc:        int = MODEL_CONFIG["mc_samples"],
                     temperature: float = 0.5,
                     ) -> Dict:
    row   = _build_row(gases, vits, equip)
    graph = build_graph(row, scaler)
    loader = DataLoader([graph], batch_size=1)
    batch  = next(iter(loader)).to(DEVICE)

    all_probs = []
    valid_paths = [p for p in FOLD_PATHS if p.exists()]
    if not valid_paths:
        raise FileNotFoundError("No fold models found — run 'python train.py' first.")

    for path in valid_paths:
        m = build_model(node_in_dim=MODEL_CONFIG["node_in_dim"]).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
        mc = m.mc_dropout_predict(batch, n_samples=n_mc)
        all_probs.append(mc["mean_probs"][0])

    stacked    = torch.stack(all_probs, dim=0)          # [K, 6]
    ens_probs  = stacked.mean(dim=0)
    log_p      = torch.log(ens_probs.clamp(min=1e-9))
    probs_cal  = torch.softmax(log_p / max(temperature, 0.01), dim=-1)
    pred_idx   = probs_cal.argmax().item()
    pred_code  = IDX_TO_CODE[pred_idx]
    pred_label = FAULT_LABELS[pred_code]

    probs_dict = {
        FAULT_LABELS[FAULT_CODES[j]]: round(float(probs_cal[j]), 4)
        for j in range(NUM_CLASSES)
    }

    # RUL
    rul_estimates = {}
    for gas, thresh in IEEE_THRESHOLDS.items():
        current = float(row["TDCG"]) if gas == "TDCG" else float(row[gas])
        vit_val = 0.0 if gas == "TDCG" else float(vits.get(gas, 0.0))
        rul_est = estimate_rul(current, vit_val, thresh, 0.0)
        rul_est["decision"] = rul_decision(rul_est["rul_months"], 0.0)
        rul_estimates[gas]  = rul_est

    cdi = compute_cdi(float(row["CO"]),   float(row["CO2"]))
    tai = compute_tai(float(row["C2H4"]), float(row["C2H6"]))
    dsi = compute_dsi(float(row["H2"]),   float(row["C2H2"]))
    ohi = compute_ohi(float(row["TDCG"]), cdi, tai, dsi)

    return {
        "equipment":      equip,
        "mode":           f"ensemble ({len(valid_paths)} folds × {n_mc} MC passes)",
        "pred_class":     pred_label,
        "pred_code":      pred_code,
        "description":    FAULT_DESC.get(pred_label, ""),
        "probabilities":  probs_dict,
        "health_indices": {
            "OHI": round(ohi, 4), "CDI": round(cdi, 4),
            "TAI": round(tai, 4), "DSI": round(dsi, 4),
        },
        "rul_estimates":  rul_estimates,
        "input_gases_ppm": {g: round(float(row[g]), 3) for g in GAS_COLS},
    }


#  Report formatting

def _fmt_rul(rul: Dict) -> str:
    m = rul["rul_months"]
    if m == float("inf"):
        return "∞"
    lo, hi = rul["lower"], rul["upper"]
    return f"{m:.1f} months  [{lo:.1f}–{hi:.1f}]"


def print_report(result: Dict) -> None:
    w = 64
    sep = "─" * w
    print(f"\n┌{sep}┐")
    print(f"│{'  PIGNN-UQ — Fault Diagnosis Report':^{w}}│")
    print(f"├{sep}┤")
    print(f"│  Equipment  : {result['equipment']:<{w-16}}│")
    pred = result['pred_class']
    desc = result['description']
    print(f"│  Prediction : {pred:<{w-16}}│")
    print(f"│  Description: {desc:<{w-16}}│")
    conf = result.get('confidence')
    unc  = result.get('uncertainty')
    if conf is not None:
        print(f"│  Confidence : {conf:<.4f}  |  Uncertainty : {unc:<.4f}  {'':<{w-44}}│")
    print(f"├{sep}┤")
    print(f"│  Class probabilities:{'':>{w-22}}│")
    probs = result['probabilities']
    row1  = "  ".join(f"{k}: {v:.4f}" for k, v in list(probs.items())[:3])
    row2  = "  ".join(f"{k}: {v:.4f}" for k, v in list(probs.items())[3:])
    print(f"│    {row1:<{w-4}}│")
    print(f"│    {row2:<{w-4}}│")
    hi = result['health_indices']
    print(f"├{sep}┤")
    print(f"│  Health indices (IEEE C57.104){'':>{w-31}}│")
    s = f"OHI={hi['OHI']:.3f}  CDI={hi['CDI']:.3f}  TAI={hi['TAI']:.3f}  DSI={hi['DSI']:.3f}"
    print(f"│    {s:<{w-4}}│")
    print(f"├{sep}┤")
    print(f"│  RUL estimates (IEEE C57.104 thresholds){'':>{w-41}}│")
    for gas, rul in result['rul_estimates'].items():
        rul_str = _fmt_rul(rul)
        dec     = rul['decision']
        line    = f"    {gas:<6} : {rul_str:<26}  → {dec}"
        print(f"│{line:<{w}}│")
    print(f"└{sep}┘\n")


#  CSV batch mode

def run_csv(csv_path: Path, output_path: Optional[Path],
            model, scaler: DGAScaler, temperature: float) -> None:
    df = pd.read_csv(csv_path)
    required = set(GAS_COLS)
    missing  = required - set(df.columns)
    if missing:
        sys.exit(f"CSV missing columns: {missing}")

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        gases = {g: float(row[g]) for g in GAS_COLS}
        vits  = {g: float(row.get(GAS_TO_VIT[g], 0.0)) for g in GAS_COLS}
        equip = str(row.get("equipement", f"sample_{i}"))
        res   = predict_single(model, scaler, gases, vits, equip,
                               temperature=temperature)
        results.append(res)
        print(f"[{i+1:03d}/{len(df)}] {equip:<20} → {res['pred_class']}  "
              f"(conf={res['confidence']:.3f}  unc={res['uncertainty']:.3f})")

    if output_path:
        output_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nResults saved to {output_path}")
    else:
        print(json.dumps(results, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PIGNN-UQ — Power transformer fault diagnosis via DGA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Gas input
    gas_group = parser.add_argument_group("DGA gas concentrations (ppm)")
    for gas in ["h2", "ch4", "c2h2", "c2h4", "c2h6", "co", "co2"]:
        gas_group.add_argument(f"--{gas}", type=float, default=0.001,
                               help=f"{gas.upper()} concentration (ppm)")

    # Degradation rates
    vit_group = parser.add_argument_group("Gas degradation rates (ppm/month) — for RUL")
    for gas in ["h2", "ch4", "c2h2", "c2h4", "c2h6", "co", "co2"]:
        vit_group.add_argument(f"--vit-{gas}", type=float, default=0.0,
                               dest=f"vit_{gas}",
                               help=f"{gas.upper()} degradation rate (ppm/month)")

    # Options
    parser.add_argument("--equip",       default="unknown",   help="Equipment identifier")
    parser.add_argument("--model",       default=str(FINAL_MODEL_PT),
                        help="Path to model weights (.pt)")
    parser.add_argument("--temperature", type=float, default=0.8919,
                        help="Temperature scaling factor (default: 0.8919 from R10)")
    parser.add_argument("--mc-samples",  type=int,   default=MODEL_CONFIG["mc_samples"],
                        help="Number of MC Dropout passes")
    parser.add_argument("--ensemble",    action="store_true",
                        help="Use all 10 fold models (ensemble, slower but more accurate)")
    parser.add_argument("--csv",         type=str, default=None,
                        help="CSV file for batch inference (columns: H2,CH4,C2H2,...)")
    parser.add_argument("--output",      type=str, default=None,
                        help="Output JSON path for CSV batch results")
    parser.add_argument("--json",        action="store_true",
                        help="Print result as JSON (for programmatic use)")

    args = parser.parse_args()

    # Build input dicts
    gases = {g.upper(): getattr(args, g) for g in ["h2", "ch4", "c2h2", "c2h4", "c2h6", "co", "co2"]}
    vits  = {g.upper(): getattr(args, f"vit_{g}") for g in ["h2", "ch4", "c2h2", "c2h4", "c2h6", "co", "co2"]}

    # Fix key mapping: argument uses C2H2 but config uses C2H2 (all upper case already)
    # "CO2" maps from "co2" arg, ensure correct mapping
    scaler = _load_or_fit_scaler()

    if args.csv:
        model_path = Path(args.model)
        if not model_path.exists():
            sys.exit(f"Model not found: {model_path}\nRun 'python train.py' first.")
        model = build_model(node_in_dim=MODEL_CONFIG["node_in_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        run_csv(Path(args.csv),
                Path(args.output) if args.output else None,
                model, scaler, args.temperature)
        return

    if args.ensemble:
        result = predict_ensemble(scaler, gases, vits, args.equip,
                                  n_mc=args.mc_samples, temperature=0.5)
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            sys.exit(f"Model not found: {model_path}\nRun 'python train.py' first.")
        model = build_model(node_in_dim=MODEL_CONFIG["node_in_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        result = predict_single(model, scaler, gases, vits, args.equip,
                                n_mc=args.mc_samples, temperature=args.temperature)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
