"""
PIGNN-UQ — DGA preprocessing pipeline
Reference: Chapter 3 — DONGMO

Author : DONGMO
GitHub : https://github.com/hashirama21
"""

from __future__ import annotations

__author__ = "DONGMO"
__github__ = "https://github.com/hashirama21"

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from utils.config import (
    BELOW_DETECTION, CODE_TO_IDX, CUMUL_COLS, DEVICE, DUVAL_COLS,
    EDGE_DEFINITIONS, FAULT_CODES, GAS_COLS, GAS_TO_VIT, GAS_VALID_RANGE,
    ID_COL, NODE_WEIGHTS, RATIO_COLS, RATIO_FORMULA, TARGET_COL, TRAIN_CONFIG,
    VIT_COLS, ZERO_REPLACEMENT,
    compute_cdi, compute_dsi, compute_edge_weight, compute_ohi, compute_tai,
    get_logger, set_seed,
)

logger = get_logger("preprocessing")


def load_dataset(path) -> pd.DataFrame:
    """
    Load and concatenate Feuil1 (154 rows) and Feuil2 (181 rows).
    Expected total: 335 samples after filtering.
    """
    logger.info(f"Loading: {path}")
    dfs = []
    for sheet in ["Feuil1", "Feuil2"]:
        try:
            df_s = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
            df_s = df_s.drop(columns=["equipement.1"], errors="ignore")
            dfs.append(df_s)
            logger.info(f"  {sheet}: {len(df_s)} rows")
        except Exception as e:
            logger.warning(f"  {sheet} skipped: {e}")

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"  Raw total: {len(df)} rows")

    n_before = len(df)
    df = df[df[TARGET_COL].isin(FAULT_CODES)].copy()
    logger.info(f"  After class filter: {len(df)} (removed {n_before - len(df)})")

    n_before = len(df)
    df = df.dropna(subset=GAS_COLS)
    logger.info(f"  After dropna on gases: {len(df)} (removed {n_before - len(df)})")

    for gas, (lo, hi) in GAS_VALID_RANGE.items():
        mask    = (df[gas] >= lo) & (df[gas] <= hi)
        removed = (~mask).sum()
        if removed:
            logger.warning(f"  {gas}: {removed} out-of-range rows removed")
        df = df[mask].copy()

    logger.info(f"  Final dataset: {len(df)} samples")
    _log_class_distribution(df)
    return df.reset_index(drop=True)


def _log_class_distribution(df: pd.DataFrame) -> None:
    labels = {201: "D1", 202: "D2", 301: "T1", 302: "T2", 303: "T3", 400: "DT"}
    for code, cnt in df[TARGET_COL].value_counts().sort_index().items():
        logger.info(f"    {labels.get(code, '?')}({code}): {cnt}")



def handle_zeros_and_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Must run BEFORE ratio computation to avoid division-by-near-zero artifacts."""
    df = df.copy()
    for gas, repl in ZERO_REPLACEMENT.items():
        df.loc[df[gas] <= BELOW_DETECTION, gas] = 0.001
        df.loc[df[gas] == 0.0, gas]             = repl

    for col in VIT_COLS:
        df[col] = df[col].fillna(0.0).clip(lower=0.0)

    logger.info("Zeros and missing values handled.")
    return df


def extract_and_validate_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute missing/invalid ratios from raw gases. Median-impute residual NaN."""
    df  = df.copy()
    eps = 1e-9

    for col, (num, den) in RATIO_FORMULA.items():
        missing = df[col].isna() | np.isinf(df[col].astype(float))
        if missing.any():
            df.loc[missing, col] = df.loc[missing, num] / (df.loc[missing, den] + eps)

    tdcg_missing = df["TDCG"].isna() | (df["TDCG"] == 0)
    if tdcg_missing.any():
        df.loc[tdcg_missing, "TDCG"] = (
            df.loc[tdcg_missing, ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO"]].sum(axis=1)
        )

    for col in RATIO_COLS + ["TDCG"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        n_nan   = df[col].isna().sum()
        if n_nan:
            df[col] = df[col].fillna(df[col].median())
            logger.warning(f"  {col}: {n_nan} NaN -> median imputed")

    logger.info("Ratios extracted and validated.")
    return df


def compute_duval_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Duval triangle coordinates.
    Must be called on each split independently to prevent data leakage.
    """
    df  = df.copy()
    eps = 1e-9
    tri = df["C2H2"] + df["C2H4"] + df["CH4"] + eps
    df["TauxC2H2"] = df["C2H2"] / tri
    df["TauxC2H4"] = df["C2H4"] / tri
    df["TauxCH4"]  = df["CH4"]  / tri
    return df


class DGAScaler:
    """
    Fitted on training data only.
    Produces 4-dimensional node features per gas:
      [0] log1p(gas) z-score normalized
      [1] log1p(gas) x physical node weight  (unnormalized)
      [2] gas rate-of-change z-score normalized
      [3] principal ratio z-score normalized
    """

    GAS_PRINCIPAL_RATIO: dict[str, str] = {
        "H2":   "CH4surH2",
        "CH4":  "CH4surH2",
        "C2H2": "C2H2surC2H4",
        "C2H4": "C2H4surC2H6",
        "C2H6": "C2H4surC2H6",
        "CO":   "CO2surCO",
        "CO2":  "CO2surCO",
    }

    def __init__(self):
        self.gas_scaler   = StandardScaler()
        self.ratio_scaler = StandardScaler()
        self.vit_scaler   = StandardScaler()
        self._fitted      = False

    def fit(self, df: pd.DataFrame) -> DGAScaler:
        self.gas_scaler.fit(np.log1p(df[GAS_COLS].values))
        self.ratio_scaler.fit(df[RATIO_COLS].values)
        self.vit_scaler.fit(df[VIT_COLS].values)
        self._fitted = True
        return self

    def transform_row(self, row: pd.Series) -> np.ndarray:
        """Transform one row into a [7, 4] node feature matrix."""
        assert self._fitted, "Scaler must be fitted before transform."

        log_raw  = np.log1p([float(row[g]) for g in GAS_COLS])
        log_norm = (log_raw - self.gas_scaler.mean_) / (self.gas_scaler.scale_ + 1e-9)
        log_w    = log_raw * np.array([NODE_WEIGHTS[g] for g in GAS_COLS])

        vit_raw  = np.array([float(row[GAS_TO_VIT[g]]) for g in GAS_COLS])
        vit_norm = (vit_raw - self.vit_scaler.mean_) / (self.vit_scaler.scale_ + 1e-9)

        ratio_idx  = [RATIO_COLS.index(self.GAS_PRINCIPAL_RATIO[g]) for g in GAS_COLS]
        ratio_raw  = np.array([float(row[RATIO_COLS[i]]) for i in ratio_idx])
        ratio_norm = ((ratio_raw - self.ratio_scaler.mean_[ratio_idx])
                      / (self.ratio_scaler.scale_[ratio_idx] + 1e-9))

        return np.stack([log_norm, log_w, vit_norm, ratio_norm], axis=1).astype(np.float32)

    def fit_transform_row(self, df: pd.DataFrame) -> DGAScaler:
        return self.fit(df)

def build_graph(row: pd.Series, scaler: DGAScaler) -> Data:
    """Build a PyG Data object for a single DGA sample."""
    node_x   = torch.tensor(scaler.transform_row(row), dtype=torch.float)
    node_idx = {gas: i for i, gas in enumerate(GAS_COLS)}

    src_list, dst_list, edge_w_list = [], [], []
    for (ni, nj, ratio_col, formula) in EDGE_DEFINITIONS:
        val = (float(row[ratio_col])
               if ratio_col in row.index
               and not pd.isna(row[ratio_col])
               and not np.isinf(float(row[ratio_col]))
               else 1.0)
        w = compute_edge_weight(val, formula)
        i, j = node_idx[ni], node_idx[nj]
        src_list    += [i, j]
        dst_list    += [j, i]
        edge_w_list += [w, w]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(edge_w_list, dtype=torch.float).unsqueeze(1)
    y          = torch.tensor([CODE_TO_IDX[int(row[TARGET_COL])]], dtype=torch.long)

    cdi = compute_cdi(float(row["CO"]),   float(row["CO2"]))
    tai = compute_tai(float(row["C2H4"]), float(row["C2H6"]))
    dsi = compute_dsi(float(row["H2"]),   float(row["C2H2"]))
    ohi = compute_ohi(float(row["TDCG"]), cdi, tai, dsi)

    return Data(
        x              = node_x,
        edge_index     = edge_index,
        edge_attr      = edge_attr,
        y              = y,
        health_indices = torch.tensor([[cdi, tai, dsi, ohi]], dtype=torch.float),
        duval          = torch.tensor(
            [[float(row["TauxC2H2"]), float(row["TauxC2H4"]), float(row["TauxCH4"])]],
            dtype=torch.float,
        ),
        vit            = torch.tensor(
            [[float(row[GAS_TO_VIT[g]]) for g in GAS_COLS]], dtype=torch.float
        ),
        equip_id       = str(row.get(ID_COL, "unknown")),
    )


def build_graph_dataset(df:         pd.DataFrame,
                        scaler:     Optional[DGAScaler] = None,
                        fit_scaler: bool = False
                        ) -> Tuple[List[Data], DGAScaler]:
    if scaler is None:
        scaler = DGAScaler()
    if fit_scaler:
        scaler.fit(df)

    graphs = []
    for idx, row in df.iterrows():
        try:
            graphs.append(build_graph(row, scaler))
        except Exception as e:
            logger.warning(f"Sample {idx} skipped: {e}")

    logger.info(f"Graphs built: {len(graphs)}")
    return graphs, scaler


def split_dataset(df:   pd.DataFrame,
                  seed: int = TRAIN_CONFIG["random_seed"]
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    labels  = df[TARGET_COL].values
    train_r = TRAIN_CONFIG["train_ratio"]
    val_r   = TRAIN_CONFIG["val_ratio"]
    test_r  = TRAIN_CONFIG["test_ratio"]

    df_train, df_temp, _, y_temp = train_test_split(
        df, labels, test_size=(1.0 - train_r), stratify=labels, random_state=seed
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=test_r / (val_r + test_r), stratify=y_temp, random_state=seed
    )
    logger.info(f"Split — Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}")
    return (df_train.reset_index(drop=True),
            df_val.reset_index(drop=True),
            df_test.reset_index(drop=True))


def get_kfold_splits(df:      pd.DataFrame,
                     n_folds: int = TRAIN_CONFIG["n_folds"],
                     seed:    int = TRAIN_CONFIG["random_seed"]
                     ) -> List[Tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    logger.info(f"{n_folds}-fold cross-validation splits generated.")
    return list(skf.split(df, df[TARGET_COL]))


def run_preprocessing(data_path
                      ) -> Tuple[List[Data], List[Data], List[Data],
                                 DGAScaler,
                                 pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end preprocessing pipeline.

    Operation order (data-leakage-free):
      1. Load + validate
      2. Zero replacement          <- before ratio computation
      3. Ratio computation         <- after zero replacement
      4. Stratified split          <- before Duval (prevents leakage)
      5. Duval coordinates         <- computed per split independently
      6. Graph construction        <- scaler fitted on train only

    Returns: (train_graphs, val_graphs, test_graphs, scaler, df_train, df_val, df_test)
    """
    df = load_dataset(data_path)
    df = handle_zeros_and_missing(df)
    df = extract_and_validate_ratios(df)

    df_train, df_val, df_test = split_dataset(df)

    df_train = compute_duval_coords(df_train)
    df_val   = compute_duval_coords(df_val)
    df_test  = compute_duval_coords(df_test)

    train_graphs, scaler = build_graph_dataset(df_train, fit_scaler=True)
    val_graphs,   _      = build_graph_dataset(df_val,   scaler=scaler)
    test_graphs,  _      = build_graph_dataset(df_test,  scaler=scaler)

    logger.info("Preprocessing complete.")
    return train_graphs, val_graphs, test_graphs, scaler, df_train, df_val, df_test


if __name__ == "__main__":
    from utils.config import DATA_PATH
    train_g, val_g, test_g, *_ = run_preprocessing(DATA_PATH)
    print(f"Train: {len(train_g)}  Val: {len(val_g)}  Test: {len(test_g)}")
    g0 = train_g[0]
    print(f"Nodes: {g0.x.shape}  Edges: {g0.edge_index.shape}  y: {g0.y}")
