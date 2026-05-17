"""
PIGNN-UQ — Model architecture
Physics-Informed Graph Attention Network with MC Dropout uncertainty quantification
Reference: Chapter 3 — DONGMO

Author : DONGMO
GitHub : https://github.com/hashirama21
"""

from __future__ import annotations

__author__ = "DONGMO"
__github__ = "https://github.com/hashirama21"

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv
from torch_geometric.nn.aggr import AttentionalAggregation

from utils.config import CLASS_WEIGHTS, DEVICE, GAS_COLS, MODEL_CONFIG, NUM_CLASSES, get_logger

logger = get_logger("model")


# ── GAT layer ─────────────────────────────────────────────────────────────────

class GATLayer(nn.Module):

    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 heads:        int   = 4,
                 dropout:      float = 0.3,
                 concat:       bool  = True):
        super().__init__()
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=1,
            concat=concat,
        )
        norm_dim  = out_channels * heads if concat else out_channels
        self.norm = nn.BatchNorm1d(norm_dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self,
                x:           torch.Tensor,
                edge_index:  torch.Tensor,
                edge_attr:   Optional[torch.Tensor] = None,
                return_attn: bool = False):
        if return_attn:
            out, (ei, aw) = self.gat(x, edge_index,
                                     edge_attr=edge_attr,
                                     return_attention_weights=True)
            return self.drop(self.norm(F.relu(out))), (ei, aw)
        out = self.gat(x, edge_index, edge_attr=edge_attr)
        return self.drop(self.norm(F.relu(out)))


# ── Physics-informed loss ─────────────────────────────────────────────────────

class PhysicsLoss(nn.Module):
    """
    Arrhenius proxy + IEC 60599 discharge constraint.
    L_phys = lambda * (L_arrhenius + L_discharge)
    """

    # Normalized activation energies — order matches GAS_COLS (Table 3.2.3)
    EA_NORM = torch.tensor([0.30, 0.45, 0.90, 0.65, 0.40, 0.20, 0.15], dtype=torch.float)

    def __init__(self, lam: float = MODEL_CONFIG["physics_lambda"]):
        super().__init__()
        self.lam = lam
        self.register_buffer("ea", self.EA_NORM)

    def forward(self,
                logits:    torch.Tensor,  # [B, 6]
                node_feat: torch.Tensor,  # [B*7, 4]
                batch_vec: torch.Tensor,  # [B*7]
                ) -> torch.Tensor:
        probs     = F.softmax(logits, dim=-1)
        gas_feats = self._extract_per_graph(node_feat, batch_vec)  # [B, 7]

        ea_dev  = self.ea.to(gas_feats.device)
        t_proxy = torch.sigmoid(
            (gas_feats * ea_dev.unsqueeze(0)).sum(dim=1) / ea_dev.sum()
        )

        # T3 only (idx=4) as Arrhenius target — not T2+T3
        arr_loss = F.mse_loss(probs[:, 4], t_proxy)

        # IEC 60599: high C2H2 (idx=2) should correlate with high P(D2) (idx=1)
        disc_loss = F.mse_loss(probs[:, 1], torch.sigmoid(gas_feats[:, 2]))

        return self.lam * (arr_loss + disc_loss)

    @staticmethod
    def _extract_per_graph(node_feat: torch.Tensor,
                           batch_vec: torch.Tensor) -> torch.Tensor:
        """[B*7, 4] -> [B, 7] using the first feature dimension (log_norm)."""
        feat     = node_feat[:, 0]
        n_graphs = int(batch_vec.max().item()) + 1
        try:
            return feat.view(n_graphs, 7)
        except RuntimeError:
            out = torch.zeros(n_graphs, 7, device=feat.device)
            for g in range(n_graphs):
                s = feat[batch_vec == g]
                out[g, :min(7, len(s))] = s[:7]
            return out


# ── Main model ────────────────────────────────────────────────────────────────

class PIGNN_UQ(nn.Module):
    """
    Physics-Informed Graph Attention Network with Uncertainty Quantification.

    Round 4: hidden_dim=96, num_heads=2, BatchNorm1d, CE loss, mixup support
    GAT1 : node_in_dim(4)      ->  96 x 2  = 192
    GAT2 : 192                 ->  96 x 2  = 192
    GAT3 : 192                 ->  96 x 1  =  96
    GlobalAttentionPooling     -> [B, 96]
    MLP  : 96 -> 48 -> num_classes
    """

    def __init__(self,
                 node_in_dim:    int   = MODEL_CONFIG["node_in_dim"],
                 hidden_dim:     int   = MODEL_CONFIG["hidden_dim"],
                 num_heads:      int   = MODEL_CONFIG["num_heads"],
                 dropout_rate:   float = MODEL_CONFIG["dropout_rate"],
                 num_classes:    int   = NUM_CLASSES,
                 physics_lambda: float = MODEL_CONFIG["physics_lambda"]):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.num_classes  = num_classes
        self.dropout_rate = dropout_rate

        self.gat1 = GATLayer(in_channels=node_in_dim,
                             out_channels=hidden_dim,
                             heads=num_heads,
                             dropout=dropout_rate,
                             concat=True)

        self.gat2 = GATLayer(in_channels=hidden_dim * num_heads,
                             out_channels=hidden_dim,
                             heads=num_heads,
                             dropout=dropout_rate,
                             concat=True)

        self.gat3 = GATLayer(in_channels=hidden_dim * num_heads,
                             out_channels=hidden_dim,
                             heads=1,
                             dropout=dropout_rate,
                             concat=False)

        self.pool = AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.physics_loss_fn = PhysicsLoss(lam=physics_lambda)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,
                data:        Data,
                return_attn: bool = False
                ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        dev        = next(self.parameters()).device
        x          = data.x.to(dev)
        edge_index = data.edge_index.to(dev)
        edge_attr  = data.edge_attr.to(dev) if data.edge_attr is not None else None
        batch      = (data.batch.to(dev) if data.batch is not None
                      else torch.zeros(x.size(0), dtype=torch.long, device=dev))

        x = self.gat1(x, edge_index, edge_attr)
        x = self.gat2(x, edge_index, edge_attr)

        attn_info = None
        if return_attn:
            x, attn_info = self.gat3(x, edge_index, edge_attr, return_attn=True)
        else:
            x = self.gat3(x, edge_index, edge_attr)

        logits = self.classifier(self.pool(x, batch))
        return logits, attn_info

    def forward_embed(self, data: Data) -> torch.Tensor:
        """Return pooled graph embedding before the classifier head."""
        dev        = next(self.parameters()).device
        x          = data.x.to(dev)
        edge_index = data.edge_index.to(dev)
        edge_attr  = data.edge_attr.to(dev) if data.edge_attr is not None else None
        batch      = (data.batch.to(dev) if data.batch is not None
                      else torch.zeros(x.size(0), dtype=torch.long, device=dev))
        x = self.gat1(x, edge_index, edge_attr)
        x = self.gat2(x, edge_index, edge_attr)
        x = self.gat3(x, edge_index, edge_attr)
        return self.pool(x, batch)  # [B, hidden_dim]

    def compute_loss(self,
                     logits:        torch.Tensor,
                     targets:       torch.Tensor,
                     data:          Data,
                     class_weights: Optional[torch.Tensor] = None
                     ) -> Dict[str, torch.Tensor]:
        dev = logits.device
        w   = class_weights.to(dev) if class_weights is not None else None
        ce  = F.cross_entropy(logits, targets, weight=w, label_smoothing=0.02)
        batch = (data.batch.to(dev) if data.batch is not None
                 else torch.zeros(data.x.size(0), dtype=torch.long, device=dev))
        phys  = self.physics_loss_fn(logits, data.x.to(dev), batch)
        return {"total": ce + phys, "ce": ce, "physics": phys}

    # ── MC Dropout inference ──────────────────────────────────────────────────

    def mc_dropout_predict(self,
                           data:      Data,
                           n_samples: int = MODEL_CONFIG["mc_samples"]
                           ) -> Dict[str, torch.Tensor]:
        """
        T stochastic forward passes with dropout active.
        Train/eval state is saved and restored after inference.
        Variance normalized by (K-1)/K^2 (theoretical softmax maximum).
        """
        was_training = self.training
        self.train()

        dev       = next(self.parameters()).device
        all_probs = []

        with torch.no_grad():
            for _ in range(n_samples):
                logits, _ = self.forward(data)
                all_probs.append(F.softmax(logits, dim=-1).unsqueeze(0))

        self.train(was_training)

        stack      = torch.cat(all_probs, dim=0)   # [T, B, K]
        mean_probs = stack.mean(dim=0)
        variance   = stack.var(dim=0).mean(dim=1)

        K           = float(self.num_classes)
        max_var     = torch.tensor((K - 1) / (K ** 2), device=dev)
        uncertainty = (variance / max_var).clamp(0.0, 1.0)

        return {
            "mean_probs":  mean_probs,
            "uncertainty": uncertainty,
            "confidence":  1.0 - uncertainty,
            "pred_class":  mean_probs.argmax(dim=1),
        }

    def mc_dropout_predict_batched(self,
                                   data:      Data,
                                   n_samples: int = MODEL_CONFIG["mc_samples"]
                                   ) -> Dict[str, torch.Tensor]:
        """
        Vectorized MC Dropout: replicates the batch T times in a single forward pass.
        ~5-10x faster than the sequential version for batch_size=1 (RUL inference).
        """
        was_training = self.training
        self.train()

        dev       = next(self.parameters()).device
        big_batch = Batch.from_data_list([data] * n_samples).to(dev)

        with torch.no_grad():
            logits_big, _ = self.forward(big_batch)  # [T*B, K]

        self.train(was_training)

        B          = logits_big.shape[0] // n_samples
        stack      = F.softmax(logits_big, dim=-1).view(n_samples, B, self.num_classes)
        mean_probs = stack.mean(dim=0)
        variance   = stack.var(dim=0).mean(dim=1)

        K           = float(self.num_classes)
        max_var     = torch.tensor((K - 1) / (K ** 2), device=dev)
        uncertainty = (variance / max_var).clamp(0.0, 1.0)

        return {
            "mean_probs":  mean_probs,
            "uncertainty": uncertainty,
            "confidence":  1.0 - uncertainty,
            "pred_class":  mean_probs.argmax(dim=1),
        }


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(node_in_dim: int = MODEL_CONFIG["node_in_dim"]) -> PIGNN_UQ:
    model    = PIGNN_UQ(node_in_dim=node_in_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"PIGNN-UQ — trainable parameters: {n_params:,}")
    return model


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_model(node_in_dim=4).to(torch.device("cpu"))

    n_edges    = 20
    x          = torch.randn(7, 4)
    edge_index = torch.stack([torch.randint(0, 7, (n_edges,)),
                               torch.randint(0, 7, (n_edges,))])
    data       = Data(x=x, edge_index=edge_index,
                      edge_attr=torch.rand(n_edges, 1),
                      y=torch.tensor([2]))
    data.batch = torch.zeros(7, dtype=torch.long)

    logits, _ = model(data)
    print(f"Logits shape : {logits.shape}")

    mc = model.mc_dropout_predict(data, n_samples=10)
    print(f"Uncertainty  : {mc['uncertainty'].item():.4f}")
    print(f"Confidence   : {mc['confidence'].item():.4f}")
    print(f"Pred class   : {mc['pred_class'].item()}")
    print(f"Training mode after MC call: {model.training}")
