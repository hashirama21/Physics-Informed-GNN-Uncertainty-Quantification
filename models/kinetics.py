"""
KG-GNN — Reaction-network topology and differentiable Arrhenius gates.

Implements Section III (Physical Foundations) and Section IV-A/B of the paper:
nodes are chemical species, directed edges are lumped pyrolysis/oxidation
channels (Table `tab:ea`), each carrying an effective activation-energy prior
E_a and a half-opening temperature T_half anchored to the IEC 60599 fault
regime boundaries.

Gate law (calibrated Arrhenius, Section III-B):
    log kappa_ij(T) = log(1/2) + (E_a^ij / R) * (1/T_half^ij - 1/T)
    kappa_ij(T)     = exp(min(log kappa, 0))  in (0, 1]

so that kappa = 0.5 exactly at T = T_half (the IEC regime boundary), the
channel is exponentially suppressed below it, and saturates open above it.
E_a is a learnable parameter confined to its published interval by the
elastic penalty Omega (Eq. `totalloss`).

Author : DONGMO
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import FAULT_CODES, FAULT_LABELS, GAS_COLS

#  Universal gas constant 
R_GAS = 8.314  # J / (mol K)

# Latent hot-spot temperature range (K): ambient PD regime up to arcing
T_MIN = 298.0
T_MAX = 1500.0

#  Reaction digraph G_chem ─
CHEM_NODES: List[str] = ["Oil", "H2", "CH4", "C2H6", "C2H4", "C2H2", "Cell", "CO", "CO2"]
NODE_IDX: Dict[str, int] = {n: i for i, n in enumerate(CHEM_NODES)}

SOURCE_NODES = ["Oil", "Cell"]
SOURCE_NODE_IDX = [NODE_IDX[n] for n in SOURCE_NODES]

# Maps the measured 7-gas vector (GAS_COLS order) onto graph node slots
GAS_NODE_IDX = [NODE_IDX[g] for g in GAS_COLS]  # H2, CH4, C2H2, C2H4, C2H6, CO, CO2


@dataclass(frozen=True)
class ReactionChannel:
    """One lumped reaction channel of Table `tab:ea`."""
    name:   str
    src:    str
    dst:    str
    ea_lo:  float   # kJ/mol — lower bound of published interval
    ea_hi:  float   # kJ/mol — upper bound
    t_half: float   # K — gate half-opening point (IEC 60599 regime boundary)


# E_a intervals: ordering/scale priors from the pyrolysis literature
# (Halstead 1973; Laidler 1961; Emsley 1994; IEEE C57.91 B=15000 K).
# T_half calibration: IEC 60599 formation-regime boundaries.
REACTION_CHANNELS: List[ReactionChannel] = [
    ReactionChannel("oil_h2",    "Oil",  "H2",   150.0, 210.0,  393.0),  # PD / low-T C-H scission
    ReactionChannel("oil_ch4",   "Oil",  "CH4",  210.0, 290.0,  473.0),  # chain cracking (T1)
    ReactionChannel("oil_c2h6",  "Oil",  "C2H6", 250.0, 306.0,  523.0),  # cracking + CH3 recomb. (T1)
    ReactionChannel("c2h6_c2h4", "C2H6", "C2H4", 216.0, 306.0,  623.0),  # dehydrogenation (T2-T3)
    ReactionChannel("ch4_c2h6",  "CH4",  "C2H6", 370.0, 430.0, 1273.0),  # methane pyrolysis route
    ReactionChannel("c2h4_c2h2", "C2H4", "C2H2", 410.0, 470.0, 1023.0),  # deep dehydrog. (arc, T3/D2)
    ReactionChannel("cell_co",   "Cell", "CO",   111.0, 125.0,  378.0),  # cellulose scission
    ReactionChannel("cell_co2",  "Cell", "CO2",  111.0, 125.0,  378.0),  # cellulose oxidation
]

NUM_EDGES = len(REACTION_CHANNELS)
EDGE_IDX: Dict[str, int] = {c.name: i for i, c in enumerate(REACTION_CHANNELS)}

EDGE_SRC = torch.tensor([NODE_IDX[c.src] for c in REACTION_CHANNELS], dtype=torch.long)
EDGE_DST = torch.tensor([NODE_IDX[c.dst] for c in REACTION_CHANNELS], dtype=torch.long)

#  Required pathway sets P_c (Eq. `physloss`) 
# A fault class is kinetically admissible only if every channel in its set is
# open at the inferred temperature. Keys follow FAULT_LABELS of the dataset.
CLASS_PATHWAYS: Dict[str, List[str]] = {
    "D1": ["oil_h2"],                    # low-energy discharge: H2 channel
    "D2": ["c2h4_c2h2"],                 # high-energy discharge: acetylene channel
    "T1": ["oil_ch4"],                   # thermal <300 C: methane channel
    "T2": ["c2h6_c2h4"],                 # thermal 300-700 C: ethylene channel
    "T3": ["c2h6_c2h4"],                 # thermal >700 C: ethylene channel (D/T3 border via C2H2)
    "DT": ["c2h6_c2h4", "c2h4_c2h2"],    # combined: thermal + arc channels
}


#  Observational consistency (L_obs) ─
# Dual of L_phys: an OPEN gate toward a gas measured near-absent is a physical
# inconsistency — the inferred temperature is too high for the observed
# signature. L_obs forces the encoder to infer the LOWEST hot-spot temperature
# consistent with the gases actually present (maximum-parsimony principle).
#
# Absence score: a_j = sigmoid((Z0 - z_j) / S) where z_j is the z-scored
# log1p concentration (node feature dim 0). z = Z0 (one std below the mean
# log-concentration) marks the soft "absent" threshold.
ABSENCE_Z0 = -1.0
ABSENCE_SCALE = 0.5

# Incoming reaction channels per measured gas (GAS_COLS order)
GAS_IN_EDGES: List[List[int]] = [
    [i for i, c in enumerate(REACTION_CHANNELS) if c.dst == g] for g in GAS_COLS
]

#  Weak IEC 60599 temperature supervision (L_T, train-time only) ─
# Hot-spot regime bounds per fault class (K), from the formation regimes of
# Table `tab:ea` / IEC 60599. Used as a soft quadratic hinge on the latent T
# of the TRUE class — weak weight, so g_phi is guided, not supervised away.
CLASS_T_RANGES: Dict[str, tuple] = {
    "D1": (298.0, 573.0),    # low-energy discharge: cold-plasma regime
    "D2": (973.0, T_MAX),    # high-energy arc
    "T1": (423.0, 573.0),    # thermal < 300 C
    "T2": (573.0, 973.0),    # thermal 300-700 C
    "T3": (973.0, T_MAX),    # thermal > 700 C
    "DT": (773.0, T_MAX),    # combined electrical + thermal
}


def build_class_t_bounds() -> tuple:
    """([C], [C]) tensors of per-class T bounds in FAULT_CODES order."""
    lo = torch.tensor([CLASS_T_RANGES[FAULT_LABELS[c]][0] for c in FAULT_CODES])
    hi = torch.tensor([CLASS_T_RANGES[FAULT_LABELS[c]][1] for c in FAULT_CODES])
    return lo, hi


def build_pathway_mask() -> torch.Tensor:
    """[C, E] boolean mask: mask[c, e] = channel e required by class c."""
    mask = torch.zeros(len(FAULT_CODES), NUM_EDGES, dtype=torch.bool)
    for ci, code in enumerate(FAULT_CODES):
        for edge_name in CLASS_PATHWAYS[FAULT_LABELS[code]]:
            mask[ci, EDGE_IDX[edge_name]] = True
    return mask


class ArrheniusGates(nn.Module):
    """
    kappa_ij(T) per Eq. `gate`, calibrated so kappa(T_half) = 1/2.

    E_a is a learnable parameter (kJ/mol) initialised at the centre of its
    published interval; `elastic_penalty` (Omega in Eq. `totalloss`) keeps it
    inside the interval so the gates retain their kJ/mol interpretation.
    """

    def __init__(self, learn_ea: bool = True):
        super().__init__()
        ea_center = torch.tensor([(c.ea_lo + c.ea_hi) / 2.0 for c in REACTION_CHANNELS])
        self.ea = nn.Parameter(ea_center, requires_grad=learn_ea)
        self.register_buffer("ea_lo", torch.tensor([c.ea_lo for c in REACTION_CHANNELS]))
        self.register_buffer("ea_hi", torch.tensor([c.ea_hi for c in REACTION_CHANNELS]))
        self.register_buffer("inv_t_half",
                             torch.tensor([1.0 / c.t_half for c in REACTION_CHANNELS]))

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        """T: [B, 1] Kelvin -> gate openings kappa: [B, E] in (0, 1]."""
        ea_joule = self.ea * 1e3                                   # kJ/mol -> J/mol
        log_k = math.log(0.5) + (ea_joule / R_GAS).unsqueeze(0) * (
            self.inv_t_half.unsqueeze(0) - 1.0 / T
        )
        return torch.exp(torch.clamp(log_k, max=0.0))

    def elastic_penalty(self) -> torch.Tensor:
        """Omega(E_a): zero inside the published interval, quadratic outside."""
        span = self.ea_hi - self.ea_lo
        over = F.relu(self.ea - self.ea_hi) / span
        under = F.relu(self.ea_lo - self.ea) / span
        return (over ** 2 + under ** 2).sum()

    def gate_map(self, T: torch.Tensor) -> Dict[str, float]:
        """Human-readable gate openings for one sample (explainability)."""
        with torch.no_grad():
            k = self.forward(T.view(1, 1)).squeeze(0)
        return {c.name: round(float(k[i]), 4) for i, c in enumerate(REACTION_CHANNELS)}
