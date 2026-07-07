"""
KG-GNN — Kinetic-Gated Graph Neural Network (Section IV of the paper).

Topology is written by the chemistry (models/kinetics.py), not learned:
9 species nodes, 8 directed reaction channels, dense [B, 9, d] computation
(no torch_geometric dependency). Every message is multiplied by an Arrhenius
gate kappa_ij(T) driven by a latent hot-spot temperature inferred from the
gas signature. Uncertainty: predictive entropy + mutual information (BALD).

Author : DONGMO
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.kinetics import (
    ABSENCE_SCALE, ABSENCE_Z0, ArrheniusGates, EDGE_DST, EDGE_SRC,
    GAS_IN_EDGES, GAS_NODE_IDX, NUM_EDGES,
    REACTION_CHANNELS, SOURCE_NODE_IDX, T_MAX, T_MIN,
    build_class_t_bounds, build_pathway_mask,
)
from utils.config import GAS_COLS, MODEL_CONFIG, NUM_CLASSES, get_logger

logger = get_logger("kg_gnn")

N_NODES = 9
N_GAS = len(GAS_COLS)


class TemperatureEncoder(nn.Module):
    """g_phi: gas signature -> T in [T_MIN, T_MAX] K (scaled sigmoid)."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        return T_MIN + (T_MAX - T_MIN) * torch.sigmoid(self.net(x_flat))  # [B, 1]


class KineticGatedLayer(nn.Module):
    """
    m_i = sum_{j->i} kappa_ji(T) * alpha_ji * W h_j
    h_i <- LayerNorm(ReLU(U h_i + m_i)), then dropout.

    The gate multiplies the message: a closed channel (kappa -> 0) transmits
    nothing regardless of what the attention alpha would prefer.
    """

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.U = nn.Linear(dim, dim)
        self.att = nn.Linear(2 * dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self,
                h:        torch.Tensor,   # [B, N, d]
                kappa:    torch.Tensor,   # [B, E]
                edge_src: torch.Tensor,   # [E]
                edge_dst: torch.Tensor,   # [E]
                ) -> torch.Tensor:
        B, N, d = h.shape
        Wh = self.W(h)
        hs, hd = Wh[:, edge_src], Wh[:, edge_dst]                       # [B, E, d]

        # Attention normalised over incoming edges of each destination node
        scores = F.leaky_relu(self.att(torch.cat([hs, hd], dim=-1))).squeeze(-1)
        e = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)  # [B, E]
        denom = h.new_zeros(B, N).index_add_(1, edge_dst, e)
        alpha = e / (denom[:, edge_dst] + 1e-9)

        msg = (kappa * alpha).unsqueeze(-1) * hs                         # [B, E, d]
        agg = h.new_zeros(B, N, d).index_add_(1, edge_dst, msg)
        return self.drop(self.norm(F.relu(self.U(h) + agg)))


class KGGNN(nn.Module):
    """
    Forward input: node feature matrix x [B, 7, 4] from the existing DGAScaler
    ([log_norm, log*weight, vit_norm, ratio_norm] per gas). Gas features are
    projected onto their species node; Oil and Cell receive learned reservoir
    embeddings. Returns logits plus the physical by-products (T, gate map).
    """

    def __init__(self,
                 node_feat_dim: int = MODEL_CONFIG["node_in_dim"],
                 hidden_dim:    int = 64,
                 num_layers:    int = 3,
                 dropout:       float = 0.10,
                 num_classes:   int = NUM_CLASSES,
                 learn_ea:      bool = True,
                 use_gates:     bool = True,
                 gate_mode:     str = "hard"):
        super().__init__()
        self.use_gates = use_gates
        self.gate_mode = gate_mode
        self.num_classes = num_classes
        # residual gating: kappa_eff = gamma + (1-gamma)*kappa — a closed channel
        # still transmits a learned floor, so a wrong early T cannot strangle
        # gradients; losses/PCR/explanations keep the raw Arrhenius kappa
        self.gate_gamma = nn.Parameter(torch.full((NUM_EDGES,), -2.0))

        self.gas_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.src_embed = nn.Parameter(torch.randn(len(SOURCE_NODE_IDX), hidden_dim) * 0.1)
        self.temp_encoder = TemperatureEncoder(N_GAS * node_feat_dim)
        self.gates = ArrheniusGates(learn_ea=learn_ea)

        self.layers = nn.ModuleList(
            [KineticGatedLayer(hidden_dim, dropout) for _ in range(num_layers)]
        )

        self.pool_att = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.register_buffer("edge_src", EDGE_SRC.clone())
        self.register_buffer("edge_dst", EDGE_DST.clone())
        self.register_buffer("gas_slots", torch.tensor(GAS_NODE_IDX, dtype=torch.long))
        self.register_buffer("src_slots", torch.tensor(SOURCE_NODE_IDX, dtype=torch.long))
        self.register_buffer("pathway_mask", build_pathway_mask())
        t_lo, t_hi = build_class_t_bounds()
        self.register_buffer("class_t_lo", t_lo)
        self.register_buffer("class_t_hi", t_hi)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.size(0)
        T = self.temp_encoder(x.flatten(1))                              # [B, 1]
        if self.use_gates:
            kappa = self.gates(T)                                        # [B, E]
        else:
            kappa = x.new_ones(B, NUM_EDGES)

        if self.use_gates and self.gate_mode == "residual":
            gamma = torch.sigmoid(self.gate_gamma).unsqueeze(0)
            kappa_mp = gamma + (1.0 - gamma) * kappa
        else:
            kappa_mp = kappa

        h = x.new_zeros(B, N_NODES, self.gas_proj.out_features)
        h[:, self.gas_slots] = self.gas_proj(x)
        h[:, self.src_slots] = self.src_embed.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            h = layer(h, kappa_mp, self.edge_src, self.edge_dst)

        h_gas = h[:, self.gas_slots]                                     # [B, 7, d]
        w = torch.softmax(self.pool_att(h_gas), dim=1)
        embed = (w * h_gas).sum(dim=1)                                   # [B, d]
        logits = self.head(embed)

        return {"logits": logits, "T": T, "kappa": kappa, "embed": embed}

    #  Losses (Eq. totalloss) 

    def consistency_loss(self,
                         probs: torch.Tensor,   # [B, C]
                         kappa: torch.Tensor,   # [B, E]
                         ) -> torch.Tensor:
        """
        L_phys (Eq. physloss): probability mass on a class whose worst required
        channel is closed at the inferred temperature is penalised quadratically.
        """
        pen = ((1.0 - kappa) ** 2).unsqueeze(1).expand(-1, self.num_classes, -1)
        worst = torch.where(self.pathway_mask.unsqueeze(0), pen,
                            torch.zeros_like(pen)).amax(dim=-1)          # [B, C]
        return (probs * worst).sum(dim=1).mean()

    def observation_loss(self,
                         x:     torch.Tensor,   # [B, 7, node_feat]
                         kappa: torch.Tensor,   # [B, E]
                         ) -> torch.Tensor:
        """
        L_obs: penalises gates open toward gases measured near-absent, the dual
        of L_phys — the encoder must infer the lowest T consistent with the
        observed signature.  L_obs = mean_j kappa_in(j) * a_j.
        """
        z = x[..., 0]                                                    # [B, 7]
        absence = torch.sigmoid((ABSENCE_Z0 - z) / ABSENCE_SCALE)
        kappa_in = torch.stack(
            [kappa[:, idxs].amax(dim=-1) for idxs in GAS_IN_EDGES], dim=1
        )                                                                # [B, 7]
        return (kappa_in * absence).mean()

    def temperature_hinge(self,
                          T: torch.Tensor,   # [B, 1]
                          y: torch.Tensor,   # [B]
                          t_scale: float = 100.0,
                          ) -> torch.Tensor:
        """
        L_T: weak quadratic hinge keeping the latent T of the TRUE class inside
        its IEC 60599 regime bounds (train-time only).
        """
        lo, hi = self.class_t_lo[y], self.class_t_hi[y]
        t = T.squeeze(-1)
        return ((F.relu(lo - t) / t_scale) ** 2
                + (F.relu(t - hi) / t_scale) ** 2).mean()

    #  MC Dropout with entropy-based uncertainty ─

    def _set_dropout_mode(self, train_mode: bool) -> None:
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train(train_mode)

    @torch.no_grad()
    def mc_dropout_predict(self,
                           x:         torch.Tensor,
                           n_samples: int = MODEL_CONFIG["mc_samples"],
                           ) -> Dict[str, torch.Tensor]:
        """
        Only Dropout layers are switched on (model stays in eval mode), so
        predictions do not depend on batch composition.
        """
        was_training = self.training
        self.eval()
        self._set_dropout_mode(True)

        stack = torch.stack(
            [F.softmax(self.forward(x)["logits"], dim=-1) for _ in range(n_samples)],
            dim=0,                                                       # [S, B, C]
        )

        self._set_dropout_mode(False)
        self.train(was_training)

        mean_probs = stack.mean(dim=0)
        log_c = math.log(self.num_classes)
        pred_entropy = -(mean_probs * torch.log(mean_probs + 1e-12)).sum(-1) / log_c
        exp_entropy = (-(stack * torch.log(stack + 1e-12)).sum(-1)).mean(0) / log_c
        mutual_info = (pred_entropy - exp_entropy).clamp(min=0.0)

        det = self.forward(x)   # deterministic pass for T and gate map
        return {
            "mean_probs":   mean_probs,
            "pred_class":   mean_probs.argmax(dim=1),
            "pred_entropy": pred_entropy,
            "aleatoric":    exp_entropy,
            "epistemic":    mutual_info,
            "T":            det["T"],
            "kappa":        det["kappa"],
        }

    #  Physical Consistency Rate (Section V-C) ─

    def pcr(self,
            pred_class: torch.Tensor,   # [B]
            kappa:      torch.Tensor,   # [B, E]
            threshold:  float = 0.5,
            ) -> torch.Tensor:
        """Fraction of predictions whose required pathway set is open at T."""
        required = self.pathway_mask[pred_class]                         # [B, E]
        open_ok = (kappa >= threshold) | ~required
        return open_ok.all(dim=-1).float().mean()

    #  Explainability by construction (Section IV-E) ─

    @torch.no_grad()
    def explain(self, x: torch.Tensor) -> Dict:
        """Pathway-level explanation for a single sample x [1, 7, 4]."""
        out = self.forward(x)
        t_kelvin = float(out["T"].squeeze())
        kappa = out["kappa"].squeeze(0)
        return {
            "T_kelvin":  round(t_kelvin, 1),
            "T_celsius": round(t_kelvin - 273.15, 1),
            "gates": {c.name: round(float(kappa[i]), 4)
                      for i, c in enumerate(REACTION_CHANNELS)},
            "open_channels": [c.name for i, c in enumerate(REACTION_CHANNELS)
                              if float(kappa[i]) >= 0.5],
        }


def build_kg_model(**kwargs) -> KGGNN:
    model = KGGNN(**kwargs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"KG-GNN — trainable parameters: {n_params:,}")
    return model


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_kg_model()
    x = torch.randn(4, N_GAS, MODEL_CONFIG["node_in_dim"])

    out = model(x)
    print(f"logits {tuple(out['logits'].shape)}  T {out['T'].squeeze(-1).tolist()}")

    mc = model.mc_dropout_predict(x, n_samples=10)
    print(f"pred {mc['pred_class'].tolist()}  "
          f"entropy {mc['pred_entropy'].round(decimals=3).tolist()}  "
          f"MI {mc['epistemic'].round(decimals=3).tolist()}")
    print(f"PCR: {model.pcr(mc['pred_class'], mc['kappa']).item():.3f}")
    print(model.explain(x[:1]))

    # Gate physics sanity: acetylene channel must be shut at low T, open at arc T
    gates_cold = model.gates(torch.tensor([[450.0]]))
    gates_arc = model.gates(torch.tensor([[1300.0]]))
    i = [c.name for c in REACTION_CHANNELS].index("c2h4_c2h2")
    print(f"kappa(C2H4->C2H2) @450K={gates_cold[0, i]:.2e}  @1300K={gates_arc[0, i]:.4f}")
