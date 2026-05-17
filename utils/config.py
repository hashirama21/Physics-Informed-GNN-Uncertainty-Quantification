"""
PIGNN-UQ — Global configuration and shared utilities
Power transformer fault diagnosis via Dissolved Gas Analysis (DGA)
Reference: Chapter 3 — DONGMO

Author : DONGMO
GitHub : https://github.com/hashirama21
"""

from __future__ import annotations

import random
import logging
from pathlib import Path

import numpy as np
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parent.parent   # project root (one level above utils/)
DATA_PATH  = _ROOT / "data" / "dataApp_all_Df.xlsx"
OUTPUT_DIR = _ROOT / "outputs"
LOG_DIR    = _ROOT / "logs"

OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

#  Dataset columns
GAS_COLS = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]

RATIO_COLS = [
    "C2H2surC2H4",
    "CH4surH2",
    "C2H4surC2H6",
    "C2H6surH2",
    "C2H6surCH4",
    "CO2surCO",
    "C2H2surCH4",
    "C2H6surC2H2",
]

# Maps ratio column → (numerator gas, denominator gas) for on-the-fly recomputation
RATIO_FORMULA: dict[str, tuple[str, str]] = {
    "C2H2surC2H4": ("C2H2", "C2H4"),
    "CH4surH2":    ("CH4",  "H2"),
    "C2H4surC2H6": ("C2H4", "C2H6"),
    "C2H6surH2":   ("C2H6", "H2"),
    "C2H6surCH4":  ("C2H6", "CH4"),
    "CO2surCO":    ("CO2",  "CO"),
    "C2H2surCH4":  ("C2H2", "CH4"),
    "C2H6surC2H2": ("C2H6", "C2H2"),
}

CUMUL_COLS = ["TDCG", "TDCGcumul", "TDCGdelta", "TDCGdeltanorm"]

VIT_COLS = ["VitC2H2", "VitH2", "VitCH4", "VitC2H4", "VitC2H6", "VitCO", "VitCO2"]

# Explicit mapping avoids index-alignment errors from zip(GAS_COLS, VIT_COLS)
GAS_TO_VIT: dict[str, str] = {
    "H2":   "VitH2",
    "CH4":  "VitCH4",
    "C2H2": "VitC2H2",
    "C2H4": "VitC2H4",
    "C2H6": "VitC2H6",
    "CO":   "VitCO",
    "CO2":  "VitCO2",
}

DUVAL_COLS = ["TauxC2H2", "TauxC2H4", "TauxCH4"]
TARGET_COL = "Classe"
ID_COL     = "equipement"

#  Fault classes
FAULT_CODES  = [201, 202, 301, 302, 303, 400]
FAULT_LABELS = {201: "D1", 202: "D2", 301: "T1", 302: "T2", 303: "T3", 400: "DT"}
CODE_TO_IDX  = {code: i for i, code in enumerate(FAULT_CODES)}
IDX_TO_CODE  = {i: code for code, i in CODE_TO_IDX.items()}
IDX_TO_LABEL = {i: FAULT_LABELS[code] for i, code in IDX_TO_CODE.items()}
NUM_CLASSES  = len(FAULT_CODES)

# Class distribution from Feuil1 + Feuil2 — total = 335 samples
_CLASS_COUNTS = {201: 55, 202: 70, 301: 68, 302: 38, 303: 95, 400: 9}
_TOTAL        = sum(_CLASS_COUNTS.values())
CLASS_WEIGHTS = torch.tensor(
    [_TOTAL / (NUM_CLASSES * _CLASS_COUNTS[c]) for c in FAULT_CODES],
    dtype=torch.float,
)

#  IEEE C57.104 thresholds
IEEE_THRESHOLDS: dict[str, float] = {
    "H2":   1000.0,
    "C2H2":    5.0,
    "C2H4":  200.0,
    "TDCG": 2000.0,
}

GAS_VALID_RANGE: dict[str, tuple[float, float]] = {
    "H2":   (0.0,  50_000.0),
    "CH4":  (0.0,  80_000.0),
    "C2H2": (0.0,  60_000.0),
    "C2H4": (0.0, 100_000.0),
    "C2H6": (0.0,  80_000.0),
    "CO":   (0.0, 120_000.0),
    "CO2":  (0.0, 100_000.0),
}

#  Zero / below-detection replacement 
ZERO_REPLACEMENT: dict[str, float] = {
    "H2":   0.01,
    "CH4":  0.01,
    "C2H2": 0.001,
    "C2H4": 0.01,
    "C2H6": 0.001,
    "CO":   0.01,
    "CO2":  0.1,
}
BELOW_DETECTION = 1e-4

#  Graph — nodes and edges
NODE_WEIGHTS: dict[str, float] = {
    "H2":   1.0,
    "CH4":  1.2,
    "C2H2": 1.5,
    "C2H4": 1.2,
    "C2H6": 0.8,
    "CO":   1.0,
    "CO2":  0.8,
}

# (node_i, node_j, ratio_column, weight_formula)
EDGE_DEFINITIONS: list[tuple[str, str, str, str]] = [
    ("H2",   "CH4",  "CH4surH2",     "log_inv"),
    ("C2H2", "C2H4", "C2H2surC2H4",  "direct_inv"),
    ("C2H4", "C2H6", "C2H4surC2H6",  "log_inv"),
    ("CO",   "CO2",  "CO2surCO",     "min_tenth"),
    ("C2H2", "CH4",  "C2H2surCH4",   "min_ten"),
    ("C2H6", "H2",   "C2H6surH2",    "log_inv"),
    ("C2H6", "CH4",  "C2H6surCH4",   "log_inv"),
    ("H2",   "C2H2", "C2H2surC2H4",  "direct_inv"),  # PD indicator — IEC 60599
    ("CO",   "CH4",  "C2H6surCH4",   "log_inv"),     # cellulose / oil coupling
    ("C2H6", "C2H2", "C2H6surC2H2",  "log_inv"),     # discharge energy proxy
]

# node_in_dim=4: [log_gas_norm, log_gas*weight, vit_gas_norm, principal_ratio_norm]
MODEL_CONFIG: dict = {
    "node_in_dim":    4,
    "num_gat_layers": 3,
    "hidden_dim":     96,
    "num_heads":      2,
    "dropout_rate":   0.15,
    "pooling":        "global_attention",
    "output_dim":     NUM_CLASSES,
    "mc_samples":     50,
    "physics_lambda": 0.01,
}

TRAIN_CONFIG: dict = {
    "optimizer":           "adamw",
    "learning_rate":       3e-4,
    "weight_decay":        1e-4,
    "batch_size":          32,
    "noise_std":           0.10,
    "drop_edge_p":         0.15,
    "mixup_alpha":         0.4,
    "num_epochs":          500,
    "early_stop_patience": 150,
    "grad_clip":           1.0,
    "train_ratio":         0.70,
    "val_ratio":           0.15,
    "test_ratio":          0.15,
    "n_folds":             10,
    "random_seed":         42,
}

EVAL_TARGETS: dict[str, float] = {
    "accuracy":  0.95,
    "precision": 0.90,
    "recall":    0.85,
    "f1":        0.90,
    "kappa":     0.85,
    "brier":     0.10,
    "ece":       0.05,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_logger(name: str = "pignn_uq") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    fh = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def compute_edge_weight(ratio_value: float, formula: str) -> float:
    """Compute edge weight using Table 3.10 formulas."""
    eps = 1e-9
    v   = float(ratio_value) if (
        not np.isnan(ratio_value) and not np.isinf(ratio_value) and ratio_value > 0
    ) else eps

    if formula == "log_inv":
        return float(1.0 / (1.0 + abs(np.log(v + eps))))
    if formula == "direct_inv":
        return float(1.0 / (1.0 + v))
    if formula == "min_tenth":
        return float(min(1.0, v / 10.0))
    if formula == "min_ten":
        return float(min(1.0, v * 10.0))
    raise ValueError(f"Unknown formula: {formula!r}")

def compute_cdi(co: float, co2: float,
                ref_co: float = 200.0, ref_co2: float = 2000.0) -> float:
    """Cellulose Degradation Index (Section 3.6.1)."""
    eps = 1e-9
    return (co / (ref_co + eps) + co2 / (ref_co2 + eps)) / 2.0


def compute_tai(c2h4: float, c2h6: float, ref_ratio: float = 2.0) -> float:
    """Thermal Aging Index (Section 3.6.2)."""
    return (c2h4 / (c2h6 + 1e-9)) / ref_ratio


def compute_dsi(h2: float, c2h2: float) -> float:
    """Discharge Severity Index (Section 3.6.3)."""
    return np.log10(1.0 + c2h2) * np.log10(1.0 + h2) / 4.0


def compute_ohi(tdcg: float, cdi: float, tai: float, dsi: float,
                tdcg_thresh: float = 2000.0) -> float:
    """Overall Health Index in [0, 1] (Section 3.6.5)."""
    raw = (
        0.4 * min(1.0, tdcg / tdcg_thresh)
        + 0.2 * min(1.0, cdi)
        + 0.2 * min(1.0, tai / 5.0)
        + 0.2 * min(1.0, dsi / 10.0)
    )
    return float(1.0 - raw)

def estimate_rul(current_gas: float, vit_gas: float,
                 threshold: float, uncertainty: float = 0.0) -> dict:
    """
    Linear RUL estimate (Section 3.7).
    RUL = (threshold - current) / rate  [months]
    Confidence interval is scaled by MC Dropout uncertainty.
    """
    if vit_gas <= 1e-9:
        return {"rul_months": float("inf"), "lower": float("inf"),
                "upper": float("inf"), "reliable": False}
    rul    = max(0.0, (threshold - current_gas) / vit_gas)
    margin = rul * max(0.0, uncertainty)
    return {
        "rul_months": round(rul, 2),
        "lower":      round(max(0.0, rul - margin), 2),
        "upper":      round(rul + margin, 2),
        "reliable":   uncertainty < 0.4,
    }


def rul_decision(rul_months: float, uncertainty: float) -> str:
    """Maintenance decision rule (Table 3.22)."""
    if rul_months == float("inf"):
        return "Annual monitoring"
    if rul_months > 60 and uncertainty < 0.20:
        return "Annual monitoring"
    if rul_months > 24:
        return "Bi-annual DGA"
    if rul_months > 12:
        return "Quarterly DGA"
    if rul_months > 6:
        return "Monthly monitoring"
    return "Immediate replacement"
