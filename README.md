# PIGNN-UQ — Pipeline de diagnostic de transformateurs

## Structure

```
pignn_uq/
├── config.py         # Configuration, hyperparamètres, utilitaires
├── preprocessing.py  # Chargement, nettoyage, construction des graphes
├── model.py          # Architecture GAT + perte physique + MC Dropout
├── train.py          # Entraînement, validation croisée, évaluation, RUL
├── requirements.txt  # Dépendances
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
# PyTorch Geometric nécessite parfois l'installation séparée :
pip install torch_geometric
```

## Lancement

```bash
# Placer dataApp_all_Df.xlsx dans le même dossier, puis :
python train.py
```

## Dataset attendu

- Fichier : `dataApp_all_Df.xlsx`
- Feuilles utilisées : Feuil1 (154 lignes) + Feuil2 (181 lignes) = 335 total
- Colonnes gaz : H2, CH4, C2H2, C2H4, C2H6, CO, CO2
- Colonnes ratios : C2H2surC2H4, CH4surH2, C2H4surC2H6, C2H6surH2, C2H6surCH4, CO2surCO, C2H2surCH4, C2H6surC2H2
- Colonnes Vit* : VitC2H2, VitH2, VitCH4, VitC2H4, VitC2H6, VitCO, VitCO2
- Colonnes Duval : TauxC2H2, TauxC2H4, TauxCH4
- Cible : Classe (201=D1, 202=D2, 301=T1, 302=T2, 303=T3, 400=DT)

## Corrections appliquées (12 problèmes)

### Bugs critiques — implémentation

| Code | Fichier | Description |
|------|---------|-------------|
| C1 | preprocessing.py | Ordre corrigé : zéros remplacés AVANT calcul des ratios |
| C2 | preprocessing.py | Data leakage : Duval calculé APRÈS le split par fold |
| C3 | preprocessing.py | DGAScaler appliqué réellement aux features de nœuds (node_in_dim=4) |
| M1 | model.py | mc_dropout_predict : état train/eval restauré après inférence |
| M2 | model.py | Normalisation variance MC : borne (K-1)/K² au lieu de 1/K |
| T1 | train.py | Entraînement final : val gardé comme vrai held-out (pas de leakage) |
| T2 | train.py | inference_with_rul : df_test aligné passé directement |

### Erreurs logiques

| Code | Fichier | Description |
|------|---------|-------------|
| M3 | model.py | PhysicsLoss : proxy Arrhenius → P(T3) seul, pas P(T2+T3) |
| M4 | model.py | EA_NORM intégré dans le calcul (n'était plus dead code) |
| C4 | config.py | Noms de colonnes réels ("sur" au lieu de "_") |
| C5 | config.py | GAS_TO_VIT : appariement explicite gaz↔Vit (pas zip()) |

### Optimisations

| Code | Fichier | Description |
|------|---------|-------------|
| T4 | train.py | class_weights passés à cross_entropy (déséquilibre : DT=9 vs T3=95) |
| M5 | model.py | node_in_dim=4 (features enrichies : log_norm, log*weight, vit, ratio) |
| T5 | train.py | MC Dropout batché pour l'inférence RUL (×5-10 plus rapide) |
| Arch | config.py | 3 arêtes supplémentaires dans le graphe (H2↔C2H2, CO↔CH4, C2H6↔C2H2) |

## Distribution des classes (dataset réel)

| Classe | Code | N  | Poids cross-entropy |
|--------|------|----|---------------------|
| D1     | 201  | 55 | 1.02 |
| D2     | 202  | 70 | 0.80 |
| T1     | 301  | 68 | 0.82 |
| T2     | 302  | 38 | 1.47 |
| T3     | 303  | 95 | 0.59 |
| DT     | 400  |  9 | 6.20 |

La classe DT (400) est très minoritaire (9 cas) — les poids sont critiques.

## Cibles d'évaluation (Table 3.23)

| Métrique | Cible |
|----------|-------|
| Accuracy | ≥ 0.95 |
| F1 macro | ≥ 0.90 |
| Kappa    | ≥ 0.85 |
| Brier    | ≤ 0.10 |
| ECE      | ≤ 0.05 |
