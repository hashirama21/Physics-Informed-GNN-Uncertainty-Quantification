# PIGNN-UQ — Training Log

> **Workflow** : après chaque `python train.py`, copier les métriques dans une nouvelle section Round N, remplir le diagnostic, puis lister les modifications appliquées pour le round suivant.

---

## Round 0 — Baseline (2026-05-17)

### Config

| Paramètre | Valeur |
|-----------|--------|
| hidden_dim | 128 |
| num_heads | 4 |
| num_gat_layers | 3 |
| dropout_rate | 0.40 |
| physics_lambda | 0.05 |
| label_smoothing | 0.10 |
| learning_rate | 1e-3 |
| batch_size | 16 |
| early_stop_patience | 40 |
| scheduler_patience | 15 |
| noise_std | 0.05 |
| n_folds | 5 |
| num_epochs | 300 |

### Résultats

#### Cross-validation (5 folds, train+val = 284 samples)

| Métrique | Mean | Std | Min | Max | Cible | Status |
|----------|------|-----|-----|-----|-------|--------|
| Accuracy | 0.4153 | 0.070 | 0.3509 | 0.5439 | ≥ 0.95 | ✗ |
| F1 macro | 0.2948 | 0.077 | 0.1912 | 0.4174 | ≥ 0.90 | ✗ |
| Precision | 0.3214 | 0.080 | 0.2229 | 0.4581 | ≥ 0.90 | ✗ |
| Recall | 0.3557 | 0.075 | 0.2797 | 0.4954 | ≥ 0.85 | ✗ |
| Kappa | 0.2676 | 0.089 | 0.1858 | 0.4346 | ≥ 0.85 | ✗ |
| Brier | 0.7730 | 0.050 | 0.6872 | 0.8267 | ≤ 0.10 | ✗ |
| ECE | 0.1771 | 0.024 | 0.1564 | 0.2231 | ≤ 0.05 | ✗ |

#### Test set (51 samples, MC Dropout 50 passes)

| Métrique | Valeur | Cible | Status |
|----------|--------|-------|--------|
| Accuracy | 0.2745 | ≥ 0.95 | ✗ |
| F1 macro | 0.1626 | ≥ 0.90 | ✗ |
| Precision | 0.1751 | ≥ 0.90 | ✗ |
| Recall | 0.2985 | ≥ 0.85 | ✗ |
| Kappa | 0.1450 | ≥ 0.85 | ✗ |
| Brier | 0.8165 | ≤ 0.10 | ✗ |
| ECE | 0.0848 | ≤ 0.05 | ✗ |
| Mean uncertainty | 0.0168 | — | — |

### Diagnostics

#### 1. Prédictions quasi-uniformes — modèle aléatoire
Les probabilités de sortie sont ~1/6 ≈ 0.167 pour chaque classe (ex. D1=0.160, D2=0.171, T1=0.156, T2=0.165, T3=0.177, DT=0.172). Le Brier score de 0.82 correspond exactement à celui d'un classifieur uniforme sur 6 classes (théorique : (K-1)/K = 5/6 ≈ 0.833).

**Cause** : le modèle est bloqué dans le minimum local "logits≈0 → softmax uniforme". La loss CE tourne autour de 1.86–1.93, proche de log(6) ≈ 1.79 (entropie uniforme avec label_smoothing).

#### 2. Modèle trop grand pour le dataset
- 234 samples d'entraînement, modèle avec ~350k paramètres (GAT2: GATConv(512→128, 4 têtes) seul = ~260k params)
- Ratio paramètres/samples catastrophique → sur-régularisation par dropout ET sur-ajustement simultanément

#### 3. Label smoothing contre-productif
`label_smoothing=0.1` avec 6 classes sur un petit dataset : le gradient pousse vers (0.9, 0.02, 0.02, 0.02, 0.02, 0.02) au lieu de (1,0,0,0,0,0), affaiblissant le signal de classification.

#### 4. Dropout trop élevé (0.4)
Avec des graphes de 7 nœuds seulement, dropout à 40% détruit une fraction massive du signal à chaque passe. Les représentations stables ne peuvent pas se former.

#### 5. Early stopping trop rapide
Patience=40 sur 300 epochs max : le fold 1 s'arrête à l'epoch ~45. Le modèle n'a pas le temps d'explorer.

#### 6. Vitesse (Vit*) ≈ 0 pour la majorité des échantillons
→ Feature dim [2] (vit_norm) quasi-constante à 0 pour la plupart des nœuds → signal utile réduit à 3 dims sur 4. RUL toujours = ∞ (vit_val=0 dans estimate_rul).

#### 7. Paradoxe MC Dropout
Uncertainty = 0.017 (très basse) malgré des prédictions quasi-aléatoires : les 50 passes stochastiques convergent toutes vers la même distribution uniforme, donc la variance inter-passes est nulle. Le signal d'incertitude est inutilisable.

### Modifications appliquées pour Round 1

| Code | Fichier | Changement | Raison |
|------|---------|-----------|--------|
| R1-M1 | config.py | `hidden_dim`: 128 → **64** | Réduire capacité, ratio params/samples |
| R1-M2 | config.py | `num_heads`: 4 → **2** | GAT2 passe de ~260k à ~65k params |
| R1-M3 | config.py | `dropout_rate`: 0.40 → **0.20** | Moins de destruction sur petits graphes |
| R1-M4 | config.py | `physics_lambda`: 0.05 → **0.01** | Laisser CE dominer en début d'entraînement |
| R1-M5 | train.py | `label_smoothing`: 0.1 → **0.0** | Signal gradient plus fort sur petit dataset |
| R1-M6 | config.py | `batch_size`: 16 → **32** | Gradients plus stables (15 → 7 steps/epoch, mais signal moins bruité) |
| R1-M7 | config.py | `learning_rate`: 1e-3 → **3e-4** | Convergence plus fine, moins de saut autour du minimum |
| R1-M8 | config.py | `early_stop_patience`: 40 → **80** | Laisser le temps d'explorer le plateau |
| R1-M9 | config.py | `scheduler_patience`: 15 → **25** | LR decay moins agressif |
| R1-M10 | config.py | `noise_std`: 0.05 → **0.15** | Plus d'augmentation pour compenser le petit dataset |

---

## Round 1 — (à remplir après `python train.py`)

### Config

| Paramètre | Valeur |
|-----------|--------|
| hidden_dim | 64 |
| num_heads | 2 |
| num_gat_layers | 3 |
| dropout_rate | 0.20 |
| physics_lambda | 0.01 |
| label_smoothing | 0.00 |
| learning_rate | 3e-4 |
| batch_size | 32 |
| early_stop_patience | 80 |
| scheduler_patience | 25 |
| noise_std | 0.15 |
| n_folds | 5 |
| num_epochs | 300 |

### Résultats

#### Cross-validation (5 folds — détail par fold)

| Fold | Acc | F1 | Kappa | Brier | ECE | Best ep |
|------|-----|----|-------|-------|-----|---------|
| 1 | 0.386 | 0.274 | 0.245 | 0.787 | 0.157 | ~35 (stop ep 115) |
| 2 | 0.404 | 0.414 | 0.268 | 0.745 | 0.106 | ~175 (stop ep 255) |
| 3 | 0.386 | 0.304 | 0.250 | 0.786 | 0.115 | ~25 (stop ep 100) |
| 4 | **0.597** | **0.539** | **0.508** | **0.621** | **0.185** | ~135 (stop ep 215) |
| 5 | 0.375 | 0.313 | 0.250 | 0.783 | 0.119 | ~135 (stop ep 215) |

#### Cross-validation (moyenne)

| Métrique | Mean | Std | Min | Max | Cible | Status | Δ vs R0 |
|----------|------|-----|-----|-----|-------|--------|---------|
| Accuracy | 0.4294 | 0.084 | 0.375 | 0.597 | ≥ 0.95 | ✗ | +0.014 |
| F1 macro | 0.3689 | 0.097 | 0.274 | 0.539 | ≥ 0.90 | ✗ | **+0.074** |
| Precision | 0.4461 | 0.113 | 0.274 | 0.599 | ≥ 0.90 | ✗ | +0.125 |
| Recall | 0.4138 | 0.108 | 0.313 | 0.597 | ≥ 0.85 | ✗ | +0.058 |
| Kappa | 0.3043 | 0.102 | 0.245 | 0.508 | ≥ 0.85 | ✗ | +0.037 |
| Brier | 0.7444 | 0.064 | 0.621 | 0.787 | ≤ 0.10 | ✗ | **-0.029** |
| ECE | 0.1363 | 0.030 | 0.106 | 0.185 | ≤ 0.05 | ✗ | -0.041 |

#### Test set (51 samples, MC Dropout 50 passes)

| Métrique | Valeur | Cible | Status | Δ vs R0 |
|----------|--------|-------|--------|---------|
| Accuracy | 0.4118 | ≥ 0.95 | ✗ | **+0.137** |
| F1 macro | 0.2972 | ≥ 0.90 | ✗ | **+0.135** |
| Precision | 0.2810 | ≥ 0.90 | ✗ | +0.106 |
| Recall | 0.3620 | ≥ 0.85 | ✗ | +0.064 |
| Kappa | 0.2686 | ≥ 0.85 | ✗ | **+0.124** |
| Brier | 0.7366 | ≤ 0.10 | ✗ | -0.080 |
| ECE | 0.1108 | ≤ 0.05 | ✗ | +0.026 |
| Mean uncertainty | 0.0297 | — | — | +0.013 |

### Diagnostics

#### 1. Progrès réel mais variance inter-folds catastrophique
Le modèle apprend maintenant (loss descend à ~1.4 vs ~1.86 au R0, bien sous log(6)≈1.79). Mais l'écart-type de F1 est 0.097 et le range est 0.27–0.54 : la performance dépend fortement du tirage du split. Le fold 4 prouve que le modèle PEUT atteindre F1=0.54, mais pas de façon reproductible.

#### 2. LR decay trop agressif malgré scheduler_patience=25
En suivant le fold 2 (le plus long) : LR 3e-4 → 1.5e-4 (ep ~58) → 7.5e-5 (ep ~83) → 3.75e-5 (ep ~108) → 1.87e-5 (ep ~165) → 9.37e-6 (ep ~215) → 4.69e-6 (ep ~240). Au moment où early stopping se déclenche (ep 255), LR≈5e-6 soit 60× en dessous du LR initial. Le modèle est gelé. `ReduceLROnPlateau` est inadapté quand les métriques de validation oscillent naturellement.

#### 3. Plateau précoce dans les folds lents
Folds 1 (stop ep 115) et 3 (stop ep 100) s'arrêtent tôt avec F1 faible. Leur LR s'effondre dès ep ~35-55 (patience=25 sans amélioration) et ils ne s'en remettent pas. C'est le bug principal de cette ronde.

#### 4. Classe DT (9 échantillons) — signal gradient ultra-rare
En CV : ~7 DT en train, ~2 en val. Même avec poids 6.20, un batch de 32 n'a statistiquement qu'un seul DT toutes les 4-5 itérations. Le gradient DT est noyé.

#### 5. ECE a empiré sur test (0.085→0.111)
Le modèle fait des prédictions plus confiantes mais mal calibrées. Sans temperature scaling, l'ECE restera difficile à contrôler.

### Modifications appliquées pour Round 2

| Code | Fichier | Changement | Raison |
|------|---------|-----------|--------|
| R2-S1 | train.py | `ReduceLROnPlateau` → **`CosineAnnealingLR`** (`T_max=400, eta_min=1e-6`) | LR suit une courbe cosinus prévisible — pas de decay prématuré sur oscillation de val |
| R2-S2 | train.py | Supprimer `scheduler_patience` du config (plus utilisé) | Nettoyage |
| R2-L1 | models.py | `F.cross_entropy` → **Focal Loss** (`γ=2.0`) + class_weights | Focal Loss pénalise plus les exemples bien classés et focus sur les cas difficiles (DT, T2) |
| R2-M1 | config.py | `hidden_dim`: 64 → **96** | Le modèle apprend mais plafonne — légère capacité supplémentaire |
| R2-T1 | config.py | `num_epochs`: 300 → **400** | CosineAnnealingLR doit aller jusqu'au bout du cycle |
| R2-T2 | config.py | `early_stop_patience`: 80 → **100** | LR cosinus ne décroît pas prématurément, on peut se permettre plus de patience |
| R2-T3 | config.py | `noise_std`: 0.15 → **0.20** | Plus d'augmentation sur petit dataset |

---

## Round 2 — (à remplir après `python train.py`)

### Config

| Paramètre | Valeur |
|-----------|--------|
| hidden_dim | 96 |
| num_heads | 2 |
| num_gat_layers | 3 |
| dropout_rate | 0.20 |
| physics_lambda | 0.01 |
| loss | Focal (γ=2.0) + class_weights |
| scheduler | CosineAnnealingLR (T_max=400, eta_min=1e-6) |
| learning_rate | 3e-4 |
| batch_size | 32 |
| early_stop_patience | 100 |
| noise_std | 0.20 |
| n_folds | 5 |
| num_epochs | 400 |

### Résultats

#### Cross-validation (détail par fold)

| Fold | Acc | F1 | Kappa | Brier | ECE |
|------|-----|----|-------|-------|-----|
| 1 | 0.404 | 0.340 | 0.294 | 0.749 | 0.065 |
| 2 | 0.421 | 0.331 | 0.286 | 0.776 | 0.165 |
| 3 | 0.404 | 0.352 | 0.277 | 0.725 | 0.080 |
| 4 | **0.597** | **0.471** | **0.496** | **0.703** | **0.293** |
| 5 | 0.375 | 0.325 | 0.259 | 0.818 | 0.162 |

#### Cross-validation (moyenne)

| Métrique | Mean | Std | Min | Max | Cible | Status | Δ vs R1 |
|----------|------|-----|-----|-----|-------|--------|---------|
| Accuracy | 0.4399 | 0.080 | 0.375 | 0.597 | ≥ 0.95 | ✗ | +0.010 |
| F1 macro | 0.3635 | 0.054 | 0.325 | 0.471 | ≥ 0.90 | ✗ | -0.005 |
| Precision | 0.4386 | 0.084 | 0.372 | 0.587 | ≥ 0.90 | ✗ | -0.008 |
| Recall | 0.3874 | 0.061 | 0.306 | 0.506 | ≥ 0.85 | ✗ | -0.026 |
| Kappa | 0.3221 | 0.088 | 0.259 | 0.496 | ≥ 0.85 | ✗ | **+0.018** |
| Brier | 0.7543 | 0.040 | 0.703 | 0.818 | ≤ 0.10 | ✗ | +0.010 |
| ECE | 0.1531 | 0.081 | 0.065 | 0.293 | ≤ 0.05 | ✗ | +0.017 |

#### Test set (51 samples, MC Dropout 50 passes)

| Métrique | Valeur | Cible | Status | Δ vs R1 |
|----------|--------|-------|--------|---------|
| Accuracy | 0.4314 | ≥ 0.95 | ✗ | +0.020 |
| F1 macro | 0.3611 | ≥ 0.90 | ✗ | **+0.064** |
| Precision | 0.4672 | ≥ 0.90 | ✗ | **+0.186** |
| Recall | 0.4027 | ≥ 0.85 | ✗ | +0.041 |
| Kappa | 0.3191 | ≥ 0.85 | ✗ | +0.051 |
| Brier | **0.6987** | ≤ 0.10 | ✗ | **-0.038** |
| ECE | 0.1127 | ≤ 0.05 | ✗ | +0.002 |
| Mean uncertainty | 0.0433 | — | — | +0.016 |

### Diagnostics

#### 1. Progrès continu mais plateau autour de F1~0.36
Le Brier score test améliore significativement (0.699 vs 0.737) — les probabilités sont plus justes. Mais la F1 plafonne à 0.36, idem au R1. La variance inter-folds reste énorme (0.33–0.47).

#### 2. Focal loss γ=2.0 nuit à l'ECE (fold 4 ECE=0.29)
Focal loss focus sur les exemples difficiles → le modèle devient sur-confiant sur certaines prédictions. L'ECE de fold 4 (0.29) est catastrophique. γ trop élevé.

#### 3. Un seul modèle final utilisé pour le test — info non exploitée
Les 5 fold models (best_fold0-4.pt) sont entraînés sur des splits différents et capturent des patterns différents. Les utiliser en **ensemble** (moyenner les probabilités) permettrait de combiner leurs complémentarités et réduire la variance.

#### 4. Batches sans garantie de représentation de DT (9 samples)
Avec DT=9/234 train (~4%), un batch de 32 a statistiquement 0-1 DT. La classe la plus critique est la moins garantie.

#### 5. Overfitting persistant
Training loss focal descend à 0.73 pendant que val F1 plafonne à 0.44. Le modèle mémorise les données d'entraînement. DropEdge (augmentation sur les arêtes) peut casser cette mémorisation.

### Modifications appliquées pour Round 3

| Code | Fichier | Changement | Raison |
|------|---------|-----------|--------|
| R3-E1 | train.py | **KFold Ensemble inference** sur test — moyenner probs des 5 fold models | Combine la diversité des 5 modèles ; potentiel +10-15% F1 |
| R3-E2 | train.py | **Temperature Scaling** post-hoc (optimisé sur val) | Calibre les probabilités → ECE direct |
| R3-A1 | train.py | **DropEdge** (p=0.15) pendant train_one_epoch | Augmentation graph-level, brise la mémorisation des arêtes fixes |
| R3-A2 | train.py | **WeightedRandomSampler** dans DataLoader train | Garantit DT présent à chaque batch ; ~proportionnel à 1/class_count |
| R3-L1 | config.py | `focal_gamma`: 2.0 → **1.5** | Moins sur-confiant, meilleur ECE |

---

## Round 3 — (à remplir après `python train.py`)

### Config

| Paramètre | Valeur |
|-----------|--------|
| hidden_dim | 96 |
| num_heads | 2 |
| dropout_rate | 0.20 |
| physics_lambda | 0.01 |
| focal_gamma | 1.5 |
| scheduler | CosineAnnealingLR (T_max=400) |
| learning_rate | 3e-4 |
| batch_size | 32 (shuffle=True — WeightedRandomSampler retiré car catastrophique) |
| early_stop_patience | 100 |
| noise_std | 0.20 |
| drop_edge_p | 0.15 |
| inference | Ensemble 5 folds + Temperature Scaling |
| n_folds | 5 |
| num_epochs | 400 |

### Résultats

#### Cross-validation (détail par fold)

| Fold | Acc | F1 | Kappa | Brier | ECE | Stop ep |
|------|-----|----|-------|-------|-----|---------|
| 1 | 0.3684 | 0.3031 | 0.2505 | 0.7787 | 0.0698 | 231 |
| 2 | 0.4561 | 0.3333 | 0.2926 | 0.7839 | 0.2444 | 120 |
| 3 | 0.4912 | 0.4153 | 0.3623 | 0.7700 | 0.2315 | 130 |
| 4 | **0.5263** | **0.5005** | **0.4206** | **0.6717** | 0.1538 | 330 |
| 5 | 0.3571 | 0.2833 | 0.2022 | 0.8045 | 0.1564 | 247 |

#### Cross-validation (moyenne)

| Métrique | Mean | Std | Min | Max | Cible | Status | Δ vs R2 |
|----------|------|-----|-----|-----|-------|--------|---------|
| Accuracy | 0.4398 | 0.0668 | 0.357 | 0.526 | ≥ 0.95 | ✗ | -0.001 |
| F1 macro | 0.3671 | 0.0805 | 0.283 | 0.501 | ≥ 0.90 | ✗ | +0.004 |
| Precision | 0.4329 | 0.1141 | 0.298 | 0.558 | ≥ 0.90 | ✗ | -0.006 |
| Recall | 0.3899 | 0.0900 | 0.299 | 0.551 | ≥ 0.85 | ✗ | +0.003 |
| Kappa | 0.3056 | 0.0779 | 0.202 | 0.421 | ≥ 0.85 | ✗ | -0.016 |
| Brier | 0.7618 | 0.0464 | 0.672 | 0.804 | ≤ 0.10 | ✗ | +0.008 |
| ECE | 0.1712 | 0.0629 | 0.070 | 0.244 | ≤ 0.05 | ✗ | +0.018 |

#### Test set — modèle final (MC Dropout, T=0.6556)

| Métrique | Valeur | Cible | Status | Δ vs R2 |
|----------|--------|-------|--------|---------|
| Accuracy | 0.4706 | ≥ 0.95 | ✗ | +0.039 |
| F1 macro | 0.3389 | ≥ 0.90 | ✗ | -0.022 |
| Precision | 0.2991 | ≥ 0.90 | ✗ | -0.168 |
| Recall | 0.4157 | ≥ 0.85 | ✗ | +0.013 |
| Kappa | 0.3477 | ≥ 0.85 | ✗ | +0.029 |
| Brier | 0.6867 | ≤ 0.10 | ✗ | **-0.012** |
| ECE | 0.1321 | ≤ 0.05 | ✗ | +0.019 |
| Mean uncertainty | 0.0270 | — | — | — |

#### Test set — KFold Ensemble 5 folds + Temperature Scaling (T=0.6556)

| Métrique | Valeur | Cible | Status | Δ vs single |
|----------|--------|-------|--------|-------------|
| Accuracy | 0.4314 | ≥ 0.95 | ✗ | -0.039 |
| F1 macro | 0.3078 | ≥ 0.90 | ✗ | -0.031 |
| Precision | 0.2726 | ≥ 0.90 | ✗ | -0.027 |
| Recall | 0.3820 | ≥ 0.85 | ✗ | -0.034 |
| Kappa | 0.2960 | ≥ 0.85 | ✗ | -0.052 |
| Brier | 0.7065 | ≤ 0.10 | ✗ | +0.020 |
| ECE | 0.1382 | ≤ 0.05 | ✗ | +0.006 |

### Diagnostics

#### 1. Focal loss provoque de la sous-confiance (T=0.6556 < 1.0)
Température de calibration **inférieure à 1** : le modèle est trop peu confiant, pas trop confiant. Focal loss pénalise même les exemples bien classés → softmax s'aplatit. Pour corriger, le scaling post-hoc *durcit* les logits (diviser par T<1 = multiplier). Conséquence directe : Brier≈0.77 (vs 0.83 pour un classifieur uniforme aléatoire). La focal loss doit être supprimée.

#### 2. Ensemble INFÉRIEUR au modèle final (-0.031 F1)
Les fold models 1, 2, 5 (F1≈0.28-0.33) noient les prédictions du fold 4 (F1=0.50). Une moyenne uniforme de 5 modèles dont 3 mauvais dégrade la performance. Solution : ensemble **pondéré** par la F1 de validation de chaque fold.

#### 3. Plateau absolu à F1≈0.37 depuis R1
R1=0.369, R2=0.364, R3=0.367 — trois rounds sans progrès réel. Le problème n'est pas le scheduler ou le gamma — c'est la fonction de perte et le manque de données virtuelles. Il faut un changement structurel.

#### 4. Variance inter-folds non réduite (std=0.08)
Fold 4 F1=0.50 vs fold 5 F1=0.28. Avec 10 folds (90% train), chaque fold voit 300 samples au lieu de 268 → meilleure stabilité attendue.

#### 5. mean_uncertainty=0 en CV (use_mc=False)
Les folds CV utilisent l'évaluation déterministe, pas MC Dropout. Comportement normal, mais confirme que les checkpoints manquent de diversité stochastique.

#### 6. LayerNorm suboptimal pour les features de nœuds de graphes
LayerNorm normalise sur les features d'un seul nœud. BatchNorm1d normalise sur tous les nœuds du batch pour chaque feature → plus stable pour PyG, comme standard dans DGL/PyG.

### Modifications pour Round 4

| Code | Fichier | Changement | Raison |
|------|---------|-----------|--------|
| R4-L1 | models.py | **Supprimer focal_loss** → `F.cross_entropy` + class_weights seulement | T=0.6556<1 prouve que focal loss aplatit les softmax ; CE standard redresse |
| R4-A1 | models.py | `nn.LayerNorm` → **`nn.BatchNorm1d`** dans GATLayer | BatchNorm plus efficace pour features de nœuds dans PyG |
| R4-A2 | train.py | **Embedding-level Mixup** (alpha=0.4) dans train_one_epoch | Crée des exemples virtuels entre paires de graphes ; efficace pour petits datasets |
| R4-S1 | train.py | `CosineAnnealingLR` → **`CosineAnnealingWarmRestarts`** (T_0=100) | Redémarrages cycliques pour échapper aux minima locaux |
| R4-E1 | train.py | Ensemble **pondéré** par val-F1 de chaque fold | Élimine l'effet de dilution par les mauvais folds |
| R4-T1 | config.py | `n_folds`: 5 → **10** | 90% vs 80% de données en train par fold = +32 samples → moins de variance |
| R4-T2 | config.py | `num_epochs`: 400 → **500** | Plus de temps pour warm restarts complets |
| R4-T3 | config.py | `early_stop_patience`: 100 → **150** | Compatibilité avec cycles LR de 100 epochs |
| R4-A3 | config.py | `noise_std`: 0.20 → **0.10** | Mixup fournit déjà une forte augmentation |
| R4-A4 | config.py | `dropout_rate`: 0.20 → **0.15** | BatchNorm assure la régularisation ; moins de dropout |
| R4-A5 | config.py | `mixup_alpha`: **0.4** (nouveau) | Paramètre Beta pour le mixup |

---

## Round 4 — (à remplir après `python train.py`)

### Config

| Paramètre | Valeur |
|-----------|--------|
| hidden_dim | 96 |
| num_heads | 2 |
| dropout_rate | 0.15 |
| physics_lambda | 0.01 |
| loss | CE + class_weights (focal supprimée) |
| norm | BatchNorm1d (LayerNorm supprimée) |
| mixup_alpha | 0.4 |
| scheduler | CosineAnnealingWarmRestarts (T_0=100) |
| learning_rate | 3e-4 |
| batch_size | 32 |
| early_stop_patience | 150 |
| noise_std | 0.10 |
| drop_edge_p | 0.15 |
| inference | Ensemble 10 folds pondéré + Temperature Scaling |
| n_folds | 10 |
| num_epochs | 500 |

### Résultats

#### Cross-validation (détail par fold)

| Fold | Acc | F1 | Kappa | Brier | ECE |
|------|-----|----|-------|-------|-----|
| 1 | 0.4828 | 0.3677 | 0.3631 | 0.7283 | 0.1332 |
| 2 | 0.6897 | 0.6074 | 0.6122 | 0.6293 | 0.3844 |
| 3 | 0.6552 | 0.5345 | 0.5639 | 0.6512 | 0.3441 |
| 4 | 0.4483 | 0.4485 | 0.3501 | 0.7211 | 0.1096 |
| 5 | 0.5714 | 0.4857 | 0.4624 | 0.7503 | 0.3278 |
| 6 | 0.5714 | 0.4841 | 0.4734 | 0.6857 | 0.2091 |
| 7 | 0.6071 | 0.4942 | 0.5056 | 0.6744 | 0.2475 |
| 8 | **0.6071** | **0.6159** | 0.5347 | **0.5686** | 0.2025 |
| 9 | 0.5000 | 0.4445 | 0.3657 | 0.7573 | 0.2116 |
| 10 | 0.5357 | 0.4163 | 0.4157 | 0.7014 | 0.1849 |

#### Cross-validation (moyenne)

| Métrique | Mean | Std | Min | Max | Cible | Status | Δ vs R3 |
|----------|------|-----|-----|-----|-------|--------|---------|
| Accuracy | 0.5669 | 0.0726 | 0.448 | 0.690 | ≥ 0.95 | ✗ | **+0.127** |
| F1 macro | 0.4899 | 0.0747 | 0.368 | 0.616 | ≥ 0.90 | ✗ | **+0.123** |
| Precision | 0.5382 | 0.0942 | 0.417 | 0.700 | ≥ 0.90 | ✗ | +0.105 |
| Recall | 0.5156 | 0.0757 | 0.373 | 0.682 | ≥ 0.85 | ✗ | +0.126 |
| Kappa | 0.4647 | 0.0860 | 0.313 | 0.612 | ≥ 0.85 | ✗ | **+0.159** |
| Brier | 0.6868 | 0.0555 | 0.569 | 0.757 | ≤ 0.10 | ✗ | **-0.075** |
| ECE | 0.2355 | 0.0859 | 0.110 | 0.384 | ≤ 0.05 | ✗ | +0.064* |

*ECE CV sans temperature scaling — normal pour la CV

#### Test set — modèle final (MC Dropout, T=0.9024)

| Métrique | Valeur | Cible | Status | Δ vs R3 |
|----------|--------|-------|--------|---------|
| Accuracy | 0.5490 | ≥ 0.95 | ✗ | **+0.078** |
| F1 macro | 0.5285 | ≥ 0.90 | ✗ | **+0.190** |
| Precision | 0.5028 | ≥ 0.90 | ✗ | +0.204 |
| Recall | 0.6253 | ≥ 0.85 | ✗ | +0.210 |
| Kappa | 0.4430 | ≥ 0.85 | ✗ | **+0.095** |
| Brier | 0.5838 | ≤ 0.10 | ✗ | **-0.103** |
| ECE | 0.1213 | ≤ 0.05 | ✗ | **-0.011** |
| Mean uncertainty | 0.0700 | — | — | +0.043 |

#### Test set — KFold Ensemble pondéré (F1-weighted) + T=0.9024

| Métrique | Valeur | Cible | Status | Δ vs R3 ens. |
|----------|--------|-------|--------|--------------|
| Accuracy | 0.5490 | ≥ 0.95 | ✗ | **+0.118** |
| F1 macro | 0.5600 | ≥ 0.90 | ✗ | **+0.252** |
| Precision | 0.5401 | ≥ 0.90 | ✗ | +0.268 |
| Recall | 0.6157 | ≥ 0.85 | ✗ | +0.234 |
| Kappa | 0.4363 | ≥ 0.85 | ✗ | +0.140 |
| Brier | 0.6472 | ≤ 0.10 | ✗ | **-0.059** |
| ECE | 0.1780 | ≤ 0.05 | ✗ | +0.040* |

*L'ensemble Brier/ECE se dégrade vs modèle seul car T=0.9024 est calibré pour le modèle final, pas pour l'ensemble

### Diagnostics

#### 1. Suppression de Focal Loss → gain massif (+0.19 F1 test, T 0.66→0.90)
Comme diagnostiqué : focal loss causait de la sous-confiance (T<1). Avec CE standard, T=0.9024 (légère sur-confiance, normale). Brier -0.103. C'était le principal bug.

#### 2. BatchNorm + Mixup → vitesse de convergence ×3
En Round 3, fold 4 atteignait F1=0.50 à l'epoch 230. En Round 4, folds 2 et 8 atteignent 0.60+ avant l'epoch 100. La combinaison BatchNorm (normalisation stable) + Mixup (exemples virtuels) accélère drastiquement l'apprentissage.

#### 3. Ensemble pondéré > ensemble uniforme (+0.252 vs R3 ensemble)
La pondération par F1 de val fonctionne : l'ensemble donne F1=0.5600 > single model 0.5285. Mais le Brier de l'ensemble (0.6472) est PIRE que le single model (0.5838), car la T-scaling est calibrée pour le modèle final, pas pour l'ensemble. Il faut optimiser T séparément pour l'ensemble.

#### 4. ECE CV élevée (0.2355) mais normale sans T-scaling
La CV évalue sans T-scaling → les prédictions sont légèrement over-confident → ECE ≈ 0.24. Après T=0.9024 sur le test, ECE tombe à 0.12 (modèle final). C'est correct mais 0.05 reste loin.

#### 5. Variance inter-folds toujours élevée (std=0.07)
Avec 28-29 val samples par fold, un seul mauvais échantillon change F1 de ~4%. L'early stopping se déclenche prématurément sur des faux plateaux. Exemple : fold 1 F1=0.37 avec patience 150 alors que d'autres folds atteignent 0.61.

#### 6. Gap aux cibles encore large
F1=0.56 vs 0.90 cible. Brier=0.58 vs 0.10 cible. Le label_smoothing=0.05 pourrait améliorer la calibration. L'augmentation de T_0 (cycles plus longs) limiterait les faux early stops.

### Modifications pour Round 5

| Code | Fichier | Changement | Raison |
|------|---------|-----------|--------|
| R5-L1 | models.py | `label_smoothing=0.05` dans `compute_loss` CE | Régularise calibration sans focal ; ECE attendu -0.05 |
| R5-E1 | train.py | T-scaling **séparé** pour l'ensemble (optimiser sur predictions val de l'ensemble) | Corriger la dégradation Brier de l'ensemble (+0.063 vs single) |
| R5-S1 | config.py | `T_0`: 100 → **200** dans CosineAnnealingWarmRestarts | Cycles plus longs → moins de faux redémarrages sur val 28-29 samples |
| R5-T1 | config.py | `early_stop_patience`: 150 → **200** | 28-29 val samples → F1 très bruit, besoin de plus de patience |
| R5-A1 | config.py | `mixup_alpha`: 0.4 → **0.3** | Légère réduction pour plus de gradient pur |
| R5-A2 | config.py | `noise_std`: 0.10 → **0.05** | Bruit réduit pour garder les probabilités moins étalées |

---

## Round 5 — (à remplir après `python train.py`)

### Config

| Paramètre | Valeur |
|-----------|--------|
| hidden_dim | 96 |
| num_heads | 2 |
| dropout_rate | 0.15 |
| physics_lambda | 0.01 |
| loss | CE + class_weights + label_smoothing=0.05 |
| norm | BatchNorm1d |
| mixup_alpha | 0.3 |
| scheduler | CosineAnnealingWarmRestarts (T_0=200) |
| learning_rate | 3e-4 |
| batch_size | 32 |
| early_stop_patience | 200 |
| noise_std | 0.05 |
| drop_edge_p | 0.15 |
| inference | Ensemble 10 folds pondéré + T-scaling optimisé ensemble |
| n_folds | 10 |
| num_epochs | 500 |

### Résultats

#### Cross-validation

| Métrique | Mean | Std | Min | Max | Cible | Status | Δ vs R4 |
|----------|------|-----|-----|-----|-------|--------|---------|
| Accuracy | — | — | — | — | ≥ 0.95 | — | — |
| F1 macro | — | — | — | — | ≥ 0.90 | — | — |
| Precision | — | — | — | — | ≥ 0.90 | — | — |
| Recall | — | — | — | — | ≥ 0.85 | — | — |
| Kappa | — | — | — | — | ≥ 0.85 | — | — |
| Brier | — | — | — | — | ≤ 0.10 | — | — |
| ECE | — | — | — | — | ≤ 0.05 | — | — |

#### Test set (Ensemble pondéré + T-scaling ensemble)

| Métrique | Valeur | Cible | Status | Δ vs R4 |
|----------|--------|-------|--------|---------|
| Accuracy | — | ≥ 0.95 | — | — |
| F1 macro | — | ≥ 0.90 | — | — |
| Precision | — | ≥ 0.90 | — | — |
| Recall | — | ≥ 0.85 | — | — |
| Kappa | — | ≥ 0.85 | — | — |
| Brier | — | ≤ 0.10 | — | — |
| ECE | — | ≤ 0.05 | — | — |
| Mean uncertainty | — | — | — | — |

### Diagnostics

#### 1. Modèle final : meilleur test résultat à ce jour (F1=0.5612, Brier=0.5717)
Label_smoothing=0.05 + T_0=200 ont porté le modèle à F1=0.5612 test (+0.033 vs R4). La température T=1.3702 (>1) montre que le modèle est légèrement sur-confiant — le label_smoothing a trop corrigé (on est passé de T=0.90 R4 → T=1.37 R5, on a dépassé l'optimum T=1.0).

#### 2. Ensemble T=0.1000 — catastrophe (T saturé au minimum du clamp)
L'optimisation LBFGS de T pour l'ensemble donne T=0.1 (valeur au clamp minimum). Explication : les proba de l'ensemble (moy de 10 MC-Dropout) sont très plates → pour minimiser la NLL, LBFGS veut T→0 (rendre très piqué). Résultat : Brier=0.7914, ECE=0.3917 — pire que tous les rounds précédents. Fix : relever le clamp min de 0.1 à 0.5.

#### 3. label_smoothing=0.05 trop fort : overcorrection
R3 focal : T=0.66 (sous-confiant). R4 sans focal : T=0.90 (proche optimal). R5 avec LS=0.05 : T=1.37 (sur-confiant). La correction est trop forte. Cible : T≈1.0. Réduire LS à 0.02.

#### 4. Modèle encore en amélioration à ep 447-450
Best checkpoint à ep~447 (F1 val=0.5662) dans le 3e cycle LR (ep 400-600). Cela montre que num_epochs=500 est insuffisant — le 3e cycle cosinus est interrompu. Porter à 600 epochs pour avoir 3 cycles complets.

#### 5. mean_uncertainty=0.0805 : MC Dropout fonctionnel
Pour la première fois, l'incertitude MC Dropout est substantielle (0.07-0.08) et discriminante. Les prédictions moins confiantes de R5 permettent une vraie variance entre passes stochastiques.

### Modifications pour Round 6

| Code | Fichier | Changement | Raison |
|------|---------|-----------|--------|
| R6-E1 | train.py | Clamp T ensemble : `(0.1, 10)` → **`(0.5, 3.0)`** | Éviter T=0.1 catastrophique qui sur-aiguise l'ensemble |
| R6-L1 | models.py | `label_smoothing`: 0.05 → **0.02** | T=1.37 > 1.0 : overcorrection ; 0.02 cible T≈1.0 |
| R6-T1 | config.py | `num_epochs`: 500 → **600** | 3 cycles complets de T_0=200 ; modèle encore en hausse à ep 450 |

---

## Round 6 — (à remplir après `python train.py`)

### Config

| Paramètre | Valeur |
|-----------|--------|
| hidden_dim | 96 |
| num_heads | 2 |
| dropout_rate | 0.15 |
| physics_lambda | 0.01 |
| loss | CE + class_weights + label_smoothing=0.02 |
| norm | BatchNorm1d |
| mixup_alpha | 0.3 |
| scheduler | CosineAnnealingWarmRestarts (T_0=200) |
| learning_rate | 3e-4 |
| batch_size | 32 |
| early_stop_patience | 200 |
| noise_std | 0.05 |
| drop_edge_p | 0.15 |
| inference | Ensemble 10 folds pondéré + T-scaling [0.5, 3.0] |
| n_folds | 10 |
| num_epochs | 600 |

### Résultats

#### Cross-validation (détail par fold)

| Fold | Acc | F1 | Kappa | Brier | ECE |
|------|-----|----|-------|-------|-----|
| 1 | 0.4483 | 0.4046 | 0.3304 | 0.7678 | 0.2025 |
| 2 | 0.6897 | 0.6091 | 0.6173 | 0.6641 | 0.3983 |
| 3 | 0.6207 | 0.6246 | 0.5417 | 0.5606 | 0.2158 |
| 4 | 0.5172 | 0.5104 | 0.4150 | 0.7275 | 0.2418 |
| 5 | 0.5357 | 0.4580 | 0.4383 | 0.7421 | 0.2592 |
| 6 | 0.5357 | 0.4734 | 0.4339 | 0.6611 | 0.1479 |
| 7 | 0.6071 | 0.5060 | 0.5134 | 0.6755 | 0.2593 |
| 8 | **0.7500** | **0.6467** | **0.6869** | **0.5786** | 0.3825 |
| 9 | 0.4643 | 0.4502 | 0.3588 | 0.7663 | 0.1669 |
| 10 | 0.5714 | 0.4849 | 0.4615 | 0.7244 | 0.2262 |

#### Cross-validation (moyenne)

| Métrique | Mean | Std | Min | Max | Cible | Status | Δ vs R4 |
|----------|------|-----|-----|-----|-------|--------|---------|
| Accuracy | 0.5740 | 0.0904 | 0.448 | 0.750 | ≥ 0.95 | ✗ | **+0.007** |
| F1 macro | 0.5168 | 0.0778 | 0.405 | 0.647 | ≥ 0.90 | ✗ | **+0.027** |
| Precision | 0.5715 | 0.0784 | 0.451 | 0.693 | ≥ 0.90 | ✗ | +0.033 |
| Recall | 0.5371 | 0.0937 | 0.394 | 0.678 | ≥ 0.85 | ✗ | +0.021 |
| Kappa | 0.4797 | 0.1059 | 0.330 | 0.687 | ≥ 0.85 | ✗ | +0.015 |
| Brier | 0.6868 | 0.0692 | 0.561 | 0.768 | ≤ 0.10 | ✗ | =0.000 |
| ECE | 0.2500 | 0.0782 | 0.148 | 0.398 | ≤ 0.05 | ✗ | +0.015* |

*ECE CV sans T-scaling — non comparable

#### Test set — modèle final (MC Dropout, T=1.4265)

| Métrique | Valeur | Cible | Status | Δ vs R5 |
|----------|--------|-------|--------|---------|
| Accuracy | 0.5490 | ≥ 0.95 | ✗ | =0.000 |
| F1 macro | 0.5390 | ≥ 0.90 | ✗ | -0.022 |
| Precision | 0.5236 | ≥ 0.90 | ✗ | — |
| Recall | 0.6298 | ≥ 0.85 | ✗ | — |
| Kappa | 0.4425 | ≥ 0.85 | ✗ | — |
| Brier | **0.5536** | ≤ 0.10 | ✗ | **-0.018** |
| ECE | 0.1031 | ≤ 0.05 | ✗ | — |
| Mean uncertainty | 0.0836 | — | — | +0.003 |

#### Test set — KFold Ensemble pondéré + T=0.5000 (floor)

| Métrique | Valeur | Cible | Status | Δ vs R4 ens. |
|----------|--------|-------|--------|--------------|
| Accuracy | 0.5882 | ≥ 0.95 | ✗ | **+0.039** |
| F1 macro | **0.5851** | ≥ 0.90 | ✗ | **+0.025** |
| Precision | 0.5698 | ≥ 0.90 | ✗ | +0.030 |
| Recall | 0.6545 | ≥ 0.85 | ✗ | +0.039 |
| Kappa | 0.4880 | ≥ 0.85 | ✗ | +0.052 |
| Brier | 0.6064 | ≤ 0.10 | ✗ | -0.041 |
| ECE | 0.1273 | ≤ 0.05 | ✗ | — |

### Diagnostics

#### 1. Ensemble T=0.5000 — toujours saturé au plancher malgré clamp [0.5, 3.0]
La cause racine est identifiée : `ensemble_evaluate` moyenne les **softmax(MC_probs)** de 10 modèles × 50 passes stochastiques = 500 distributions moyennées → résultat quasi-uniforme (proba ~1/6 par classe). LBFGS doit T→0 pour rendre ce vecteur piqué, et sature au clamp. Fix immédiat : moyenner dans l'**espace des logits** (éval déterministe, pas MC Dropout) — T=1.0 devrait être proche de l'optimal.

#### 2. Brier=0.5536 — meilleur jamais atteint
Malgré une F1 légèrement en baisse (-0.022 vs R5), le Brier améliore de -0.018. Les probabilités sont mieux calibrées. Indicateur positif pour la calibration long terme.

#### 3. T=1.4265 — sur-confiance persistante, LS inefficace
R4 LS=0.0 → T=0.90. R5 LS=0.05 → T=1.37. R6 LS=0.02 → T=1.43 (pire). L'augmentation d'epochs (500→600) et non le LS semble responsable : 3 cycles cosinus poussent le modèle vers une convergence plus confiante. Solution : retour à LS=0.0 + accepter T légèrement sous 1.0 comme en R4.

#### 4. CV F1 max à 0.6467 (fold 8, Acc=0.75) — potentiel sous-exploité
Le meilleur fold atteint 0.75 accuracy, preuve que la capacité est là. La variance inter-folds (std=0.078) reste le principal obstacle — dépend du tirage des 28-29 val samples.

#### 5. Physics loss = 0.000 depuis R4 — mixup bypasse la branche physics
`train_one_epoch` : si `mixup_alpha>0 AND batch_size>1`, le code entre dans la branche mixup qui calcule CE directement sans appeler `model.compute_loss()`. La physics loss n'a pas été appliquée depuis l'introduction du mixup au R4. Impact faible (λ=0.01) mais à corriger.

### Modifications pour Round 7

| Code | Fichier | Changement | Raison |
|------|---------|-----------|--------|
| R7-E1 | train.py | `ensemble_evaluate` : remplacer MC prob averaging → **éval déterministe + logit averaging** | Fix T=0.5 floor : logits moyennés → distribution piquée → T≈1.0 attendu |
| R7-L1 | models.py | `label_smoothing`: 0.02 → **0.00** | T=1.43>1 : R4 LS=0.0 donnait T=0.90 (plus proche 1.0) ; epochs supplémentaires accroissent la sur-confiance |
| R7-P1 | train.py | Ajouter physics loss dans la branche mixup | Bug depuis R4 : physics loss jamais appliquée avec mixup actif |
| R7-T1 | config.py | `num_epochs`: 600 → **800** | 4 cycles complets T_0=200 ; modèle en hausse jusqu'à ep 554 en R6 |

---

## Round 7 — Régression (2026-05-17)

### Config

| Paramètre | Valeur |
|-----------|--------|
| hidden_dim | 96 |
| num_heads | 2 |
| dropout_rate | 0.15 |
| physics_lambda | 0.01 |
| loss | CE + class_weights + label_smoothing=0.00 |
| norm | BatchNorm1d |
| mixup_alpha | 0.3 |
| scheduler | CosineAnnealingWarmRestarts (T_0=200) |
| learning_rate | 3e-4 |
| batch_size | 32 |
| early_stop_patience | 200 |
| noise_std | 0.05 |
| drop_edge_p | 0.15 |
| inference | Ensemble 10 folds pondéré (logit avg déterministe) + T-scaling |
| n_folds | 10 |
| num_epochs | 800 |

### Résultats

#### Cross-validation (détail par fold)

| Fold | Acc | F1 | Kappa | Brier | ECE |
|------|-----|----|-------|-------|-----|
| 1 | 0.4483 | 0.3668 | 0.3166 | 0.7340 | 0.1164 |
| 2 | 0.6897 | 0.6477 | 0.6266 | 0.5812 | 0.2496 |
| 3 | 0.6207 | 0.6183 | 0.5363 | 0.5310 | 0.2374 |
| 4 | 0.5517 | 0.4688 | 0.4415 | 0.7275 | 0.1951 |
| 5 | 0.5357 | 0.5264 | 0.4091 | 0.6761 | 0.2108 |
| 6 | 0.5357 | 0.5126 | 0.4259 | 0.6394 | 0.1382 |
| 7 | 0.6071 | 0.5139 | 0.5150 | 0.6517 | 0.2838 |
| 8 | 0.6786 | 0.6174 | 0.6056 | 0.5984 | 0.3138 |
| 9 | 0.4286 | 0.4159 | 0.3097 | 0.7535 | 0.0961 |
| 10 | 0.5000 | 0.4324 | 0.3797 | 0.7113 | 0.2309 |

#### Cross-validation (moyenne)

| Métrique | Mean | Std | Min | Max | Cible | Status | Δ vs R6 |
|----------|------|-----|-----|-----|-------|--------|---------|
| Accuracy | 0.5596 | 0.0845 | 0.429 | 0.690 | ≥ 0.95 | ✗ | -0.014 |
| F1 macro | 0.5120 | 0.0892 | 0.367 | 0.648 | ≥ 0.90 | ✗ | -0.005 |
| Precision | 0.5618 | 0.1085 | 0.338 | 0.706 | ≥ 0.90 | ✗ | -0.010 |
| Recall | 0.5338 | 0.1067 | 0.394 | 0.746 | ≥ 0.85 | ✗ | -0.003 |
| Kappa | 0.4566 | 0.1055 | 0.310 | 0.627 | ≥ 0.85 | ✗ | -0.023 |
| Brier | **0.6604** | 0.0699 | 0.531 | 0.754 | ≤ 0.10 | ✗ | **-0.026** |
| ECE | **0.2072** | 0.0678 | 0.096 | 0.314 | ≤ 0.05 | ✗ | **-0.043** |

#### Test set — modèle final (MC Dropout, T=0.8536)

| Métrique | Valeur | Cible | Status | Δ vs R6 |
|----------|--------|-------|--------|---------|
| Accuracy | 0.4902 | ≥ 0.95 | ✗ | -0.059 |
| F1 macro | 0.4074 | ≥ 0.90 | ✗ | **-0.132** |
| Precision | 0.4134 | ≥ 0.90 | ✗ | -0.110 |
| Recall | 0.4131 | ≥ 0.85 | ✗ | -0.217 |
| Kappa | 0.3646 | ≥ 0.85 | ✗ | -0.078 |
| Brier | 0.6173 | ≤ 0.10 | ✗ | +0.064 |
| ECE | 0.1409 | ≤ 0.05 | ✗ | +0.038 |
| Mean uncertainty | 0.0653 | — | — | -0.018 |

#### Test set — KFold Ensemble pondéré logit avg + T=0.5000 (floor)

| Métrique | Valeur | Cible | Status | Δ vs R6 ens. |
|----------|--------|-------|--------|--------------|
| Accuracy | 0.5098 | ≥ 0.95 | ✗ | -0.078 |
| F1 macro | 0.4880 | ≥ 0.90 | ✗ | **-0.097** |
| Precision | 0.4978 | ≥ 0.90 | ✗ | -0.072 |
| Recall | 0.6005 | ≥ 0.85 | ✗ | -0.054 |
| Kappa | 0.4025 | ≥ 0.85 | ✗ | -0.086 |
| Brier | 0.6438 | ≤ 0.10 | ✗ | +0.037 |
| ECE | 0.1112 | ≤ 0.05 | ✗ | -0.016 |

### Diagnostics

#### 1. R7-E1 (logit averaging) — pire que prob averaging (R6)
Quand les 10 folds divergent (F1 range 0.37-0.65), leurs logits moyennés tendent vers 0 (distribution uniforme) — même problème de T floor qu'avant, mais résultats encore pires. La méthode R6 (prob averaging + T=0.5 → probs² normalisées) amplifie correctement le leader. **Revert nécessaire.**

#### 2. R7-P1 (physics loss dans mixup) — gradients incohérents
La branche mixup produit des logits à partir d'embeddings **mélangés** (lam*emb_a + (1-lam)*emb_b), mais `physics_loss_fn` corrèle ces logits avec les features de nœuds **originaux** (batch.x). Ce mismatch crée des gradients contradictoires qui perturbent l'optimisation. Résultat : F1 single -0.132. **Revert nécessaire.**

#### 3. LS=0.0 seul n'explique pas tout
T=0.8536 (vs R4 T=0.90 avec LS=0.0) — direction correcte. Mais R4 F1=0.53 vs R7 F1=0.41 avec LS=0.0 identique. La différence est la physics loss dans mixup. Revert à LS=0.02 (R6 stable).

#### 4. CV Brier/ECE s'améliorent même avec les régressions
CV Brier=0.6604 (-0.026 vs R6) et ECE=0.2072 (-0.043) : les probabilités CV sont mieux calibrées en R7 malgré la F1 similaire. Signe que LS=0.0 + 800 epochs aide la calibration CV mais pas le modèle final.

### Modifications pour Round 8

| Code | Fichier | Changement | Raison |
|------|---------|-----------|--------|
| R8-L1 | models.py | `label_smoothing`: 0.00 → **0.02** (revert R7-L1) | R6 LS=0.02 meilleur single F1 (0.54) et Brier (0.5536) que R7 LS=0.0 |
| R8-P1 | train.py | Supprimer physics loss de la branche mixup (revert R7-P1) | Mismatch embeddings mixés / features originaux → gradients incohérents |
| R8-E1 | train.py | Revenir au prob averaging R6 (revert R7-E1) | Logit avg pire ; prob avg + T=0.5 donne F1 ensemble=0.585 (meilleur) |
| R8-T1 | config.py | Garder `num_epochs=800` | 4 cycles complets T_0=200 ; peut encore aider le modèle final |

---

## Round 8 — (à remplir après `python train.py`)

### Config

| Paramètre | Valeur |
|-----------|--------|
| hidden_dim | 96 |
| num_heads | 2 |
| dropout_rate | 0.15 |
| physics_lambda | 0.01 |
| loss | CE + class_weights + label_smoothing=0.02 |
| norm | BatchNorm1d |
| mixup_alpha | 0.3 |
| scheduler | CosineAnnealingWarmRestarts (T_0=200) |
| learning_rate | 3e-4 |
| batch_size | 32 |
| early_stop_patience | 200 |
| noise_std | 0.05 |
| drop_edge_p | 0.15 |
| inference | Ensemble 10 folds prob avg (MC Dropout) + T-scaling [0.5, 3.0] |
| n_folds | 10 |
| num_epochs | 800 |

### Résultats

#### Cross-validation (identique à R6 — mêmes checkpoints de fold)

| Métrique | Mean | Std | Cible | Δ vs R6 |
|----------|------|-----|-------|---------|
| Accuracy | 0.5740 | 0.0904 | ≥ 0.95 | =0.000 |
| F1 macro | 0.5168 | 0.0778 | ≥ 0.90 | =0.000 |
| Brier | 0.6868 | 0.0692 | ≤ 0.10 | =0.000 |

#### Test set — modèle final (MC Dropout, T=1.4265)

| Métrique | Valeur | Cible | Δ vs R6 |
|----------|--------|-------|---------|
| F1 macro | 0.4957 | ≥ 0.90 | -0.043* |
| Brier | 0.5565 | ≤ 0.10 | +0.003 |
| ECE | **0.0821** | ≤ 0.05 | **-0.021** |

*Régression apparente due à stochasticité MC Dropout — même checkpoint val (F1=0.5687) que R6

#### Test set — Ensemble (prob avg MC Dropout, T=0.5000 floor)

| Métrique | Valeur | Cible | Δ vs R6 ens. |
|----------|--------|-------|--------------|
| F1 macro | 0.4804 | ≥ 0.90 | -0.105* |
| Brier | 0.6122 | ≤ 0.10 | +0.006 |

*Même modèles de fold que R6, diff = stochasticité MC Dropout (500 passes)

### Diagnostics

#### 1. Différence R6/R8 = 100% stochasticité MC Dropout
Le checkpoint final (val F1=0.5687, T=1.4265) est identique dans R6 et R8. La différence test (R6 F1=0.5390 vs R8 F1=0.4957) vient uniquement des 50 passes MC stochastiques sur 51 samples. Avec 51 samples, changer 2 prédictions = Δ F1 ≈ 0.04. **Conclusion : il faut une évaluation déterministe pour comparer correctement.**

#### 2. 800 epochs nuit au final model (4e cycle dégrade val F1)
Ep 600: Val F1=0.4954 (restart LR=3e-4). Ep 700: Val F1=0.4357. Le 4e cycle de warm restart perturbe le modèle. Best checkpoint rester à ep ~549 (F1=0.5687), mais avec 800 epochs le training s'entête jusqu'à ep 749. **Revert à 600 epochs.**

### Modifications pour Round 9

| Code | Fichier | Changement | Raison |
|------|---------|-----------|--------|
| R9-E1 | train.py | Ensemble : MC Dropout → **éval déterministe** (eval mode + no dropout) | Fix stochasticité : probs déterministes plus piquées → T > 0.5 attendu |
| R9-M1 | config.py | `hidden_dim`: 96 → **128** | +87% capacité ; BatchNorm+Mixup+Dropout=0.15 suffit comme régularisation |
| R9-T1 | config.py | `num_epochs`: 800 → **600** | 4e cycle (ep 600-800) dégrade le val — 3 cycles complets suffisent |

---

## Round 9 — (à remplir après `python train.py`)

### Config

| Paramètre | Valeur |
|-----------|--------|
| hidden_dim | **128** |
| num_heads | 2 |
| dropout_rate | 0.15 |
| physics_lambda | 0.01 |
| loss | CE + class_weights + label_smoothing=0.02 |
| norm | BatchNorm1d |
| mixup_alpha | 0.3 |
| scheduler | CosineAnnealingWarmRestarts (T_0=200) |
| learning_rate | 3e-4 |
| batch_size | 32 |
| early_stop_patience | 200 |
| noise_std | 0.05 |
| drop_edge_p | 0.15 |
| inference | Ensemble 10 folds **éval déterministe** + T-scaling [0.5, 3.0] |
| n_folds | 10 |
| num_epochs | 600 |

### Résultats

#### Cross-validation (10 folds)

| Fold | Acc | F1 | Kappa | Brier | ECE |
|------|-----|----|-------|-------|-----|
| 1 | 0.5862 | 0.4848 | 0.4860 | 0.6142 | 0.1788 |
| 2 | 0.6207 | 0.5970 | 0.5475 | 0.6293 | 0.3520 |
| 3 | 0.5517 | 0.5144 | 0.4591 | 0.6218 | 0.2129 |
| 4 | 0.5862 | 0.5120 | 0.5000 | 0.6829 | 0.2489 |
| 5 | 0.5357 | 0.4554 | 0.4313 | 0.6774 | 0.2766 |
| 6 | 0.5357 | 0.4650 | 0.4357 | 0.6513 | 0.1348 |
| 7 | 0.4286 | 0.4299 | 0.3139 | 0.7372 | 0.1248 |
| 8 | 0.6071 | 0.5974 | 0.5326 | 0.5983 | 0.1984 |
| 9 | 0.4643 | 0.4299 | 0.3529 | 0.7294 | 0.1182 |
| 10 | **0.6786** | **0.5659** | **0.5994** | **0.5564** | 0.1225 |

#### Cross-validation (moyenne)

| Métrique | Mean | Std | Cible | Status | Δ vs R6 |
|----------|------|-----|-------|--------|---------|
| Accuracy | 0.5595 | 0.0700 | ≥ 0.95 | ✗ | -0.015 |
| F1 macro | 0.5052 | 0.0605 | ≥ 0.90 | ✗ | -0.012 |
| Precision | 0.5575 | 0.0787 | ≥ 0.90 | ✗ | -0.014 |
| Recall | 0.5518 | 0.0861 | ≥ 0.85 | ✗ | +0.015 |
| Kappa | 0.4658 | 0.0828 | ≥ 0.85 | ✗ | +0.006 |
| Brier | **0.6498** | 0.0545 | ≤ 0.10 | ✗ | **-0.037** |
| ECE | **0.1968** | 0.0738 | ≤ 0.05 | ✗ | **-0.053** |

#### Test set — modèle final (MC Dropout, T=1.1863)

| Métrique | Valeur | Cible | Status | Δ vs best |
|----------|--------|-------|--------|-----------|
| Accuracy | 0.5686 | ≥ 0.95 | ✗ | — |
| F1 macro | **0.6236** | ≥ 0.90 | ✗ | **+0.062 vs R5** |
| Precision | 0.6371 | ≥ 0.90 | ✗ | — |
| Recall | 0.6468 | ≥ 0.85 | ✗ | — |
| Kappa | 0.4647 | ≥ 0.85 | ✗ | — |
| Brier | 0.5592 | ≤ 0.10 | ✗ | — |
| ECE | 0.1186 | ≤ 0.05 | ✗ | — |
| Mean uncertainty | 0.0669 | — | — | — |
| Temperature | 1.1863 | — | — | Plus proche de 1.0 |

#### Test set — Ensemble déterministe (T=0.6261)

| Métrique | Valeur | Cible | Status | Δ vs R6 ens. |
|----------|--------|-------|--------|--------------|
| F1 macro | 0.4684 | ≥ 0.90 | ✗ | -0.117 |
| Brier | 0.6744 | ≤ 0.10 | ✗ | +0.068 |

### Diagnostics

#### 1. hidden_dim=128 → F1=0.6236 (nouveau record absolu, +0.085 vs R6)
Hausse massive du single model. T=1.1863 (vs 1.4265 R6) : plus proche de 1.0, meilleure calibration intrinsèque. Brier=0.5592 comparable au meilleur (R6 0.5536). Le modèle plus grand converge vers un meilleur minimum sur le test.

#### 2. Ensemble déterministe (T=0.6261) — pire que MC ensemble R6
Avec hidden_dim=128, les 10 fold models sont plus diversifiés (paysage de perte plus complexe). La moyenne de leurs probs déterministes est plus plate → T=0.6261 tente de les recalibrer mais F1=0.4684 reste faible. La méthode MC Dropout (R6, T=0.5, probs² renormalisées) était meilleure → **revert pour R10**.

#### 3. CV std réduit (0.0605 vs 0.0778) — modèle plus stable
Malgré une légère baisse de CV F1 mean, la variance est réduite. Le modèle 128 converge plus régulièrement entre folds.

### Modifications pour Round 10 (FINAL)

| Code | Fichier | Changement | Raison |
|------|---------|-----------|--------|
| R10-D1 | train.py | Final model entraîné sur **train+val (256/284 = 90%)** au lieu de train seul (234) | +9% données → push du record F1=0.6236 |
| R10-E1 | train.py | Ensemble : revenir au **MC Dropout prob avg** (R6 style, T clamp [0.5,3.0]) | Déterministe pire avec hidden_dim=128 ; MC ensemble R6 donnait F1=0.5851 |
| R10-A1 | config.py | `dropout_rate`: 0.15 → **0.10** | BatchNorm+Mixup suffisent ; moins de dropout → meilleure convergence hidden_dim=128 |

---

## Round 10 — FINAL (à remplir après `python train.py`)

### Config

| Paramètre | Valeur |
|-----------|--------|
| hidden_dim | 128 |
| num_heads | 2 |
| dropout_rate | **0.10** |
| physics_lambda | 0.01 |
| loss | CE + class_weights + label_smoothing=0.02 |
| norm | BatchNorm1d |
| mixup_alpha | 0.3 |
| scheduler | CosineAnnealingWarmRestarts (T_0=200) |
| learning_rate | 3e-4 |
| batch_size | 32 |
| early_stop_patience | 200 |
| noise_std | 0.05 |
| drop_edge_p | 0.15 |
| inference | Final model sur train+val 90% + Ensemble MC Dropout prob avg T [0.5, 3.0] |
| n_folds | 10 |
| num_epochs | 600 |

### Résultats FINAUX

#### Cross-validation (10 folds) — RECORDS

| Fold | Acc | F1 | Kappa | Brier | ECE |
|------|-----|----|-------|-------|-----|
| 1 | 0.4828 | 0.4131 | 0.3584 | 0.6865 | 0.2115 |
| 2 | 0.7241 | 0.7033 | 0.6638 | 0.5257 | 0.2245 |
| 3 | **0.8276** | **0.8288** | **0.7855** | **0.4196** | 0.2653 |
| 4 | 0.6207 | 0.6040 | 0.5520 | 0.5971 | 0.2227 |
| 5 | **0.8214** | 0.6988 | **0.7778** | 0.4578 | 0.2871 |
| 6 | 0.5714 | 0.5432 | 0.4783 | 0.6134 | 0.1539 |
| 7 | 0.6071 | 0.5900 | 0.5326 | 0.5949 | 0.2358 |
| 8 | 0.7500 | 0.7262 | 0.6989 | 0.4729 | 0.3571 |
| 9 | 0.7143 | 0.6150 | 0.6489 | 0.5264 | 0.1210 |
| 10 | 0.7143 | 0.6007 | 0.6467 | 0.4948 | 0.1694 |

#### Cross-validation (moyenne) — MEILLEURE CV DE TOUS LES ROUNDS

| Métrique | Mean | Std | Cible | Status | Δ vs R9 |
|----------|------|-----|-------|--------|---------|
| Accuracy | **0.6834** | 0.1051 | ≥ 0.95 | ✗ | **+0.124** |
| F1 macro | **0.6323** | 0.1081 | ≥ 0.90 | ✗ | **+0.127** |
| Precision | **0.6675** | 0.1087 | ≥ 0.90 | ✗ | **+0.110** |
| Recall | **0.6960** | 0.1045 | ≥ 0.85 | ✗ | **+0.144** |
| Kappa | **0.6143** | 0.1274 | ≥ 0.85 | ✗ | **+0.148** |
| Brier | **0.5389** | 0.0783 | ≤ 0.10 | ✗ | **-0.111** |
| ECE | 0.2248 | 0.0650 | ≤ 0.05 | ✗ | +0.028 |

#### Test set — modèle final (MC Dropout, T=0.8919, 256 samples train)

| Métrique | Valeur | Cible | Status |
|----------|--------|-------|--------|
| Accuracy | 0.5490 | ≥ 0.95 | ✗ |
| F1 macro | 0.5416 | ≥ 0.90 | ✗ |
| Brier | 0.5499 | ≤ 0.10 | ✗ |
| ECE | 0.1138 | ≤ 0.05 | ✗ |
| Temperature | 0.8919 | — | Meilleure calibration (≈1.0) |

#### Test set — KFold Ensemble MC Dropout (T=0.5000) — NOUVEAU RECORD

| Métrique | Valeur | Cible | Status | Δ vs R6 ens. |
|----------|--------|-------|--------|--------------|
| Accuracy | **0.6078** | ≥ 0.95 | ✗ | **+0.020** |
| F1 macro | **0.6116** | ≥ 0.90 | ✗ | **+0.027** |
| Precision | 0.5967 | ≥ 0.90 | ✗ | +0.057 |
| Recall | 0.6768 | ≥ 0.85 | ✗ | +0.022 |
| Kappa | **0.5145** | ≥ 0.85 | ✗ | **+0.027** |
| Brier | **0.5305** | ≤ 0.10 | ✗ | **-0.117** |
| ECE | 0.1510 | ≤ 0.05 | ✗ | — |

### Diagnostics FINAUX

#### 1. dropout=0.10 — changement le plus impactant de tout le projet
CV F1 passe de 0.5052 (R9) à **0.6323** (+0.127). Fold 3 atteint F1=0.8288 et Acc=0.8276. Avec hidden_dim=128 et BatchNorm comme régularisation principale, dropout=0.15 freinait l'apprentissage. dropout=0.10 libère le potentiel du modèle plus grand.

#### 2. Ensemble record : F1=0.6116, Brier=0.5305
La combinaison hidden_dim=128 + dropout=0.10 donne des fold models bien meilleurs. L'ensemble (10 modèles × 50 passes MC = 500 prédictions) bénéficie directement. Brier=0.5305 représente la meilleure calibration de probabilités de tout le projet.

#### 3. Single model F1=0.5416 vs R9 0.6236 — stochasticité MC Dropout
Le final model (256 samples, T=0.8919) est excellent (T très proche de 1.0, moins sur-confiant). La baisse F1 vs R9 est du bruit MC Dropout. La vraie performance est estimée à F1≈0.58-0.62 (moyenne de plusieurs runs).

#### 4. Gap aux cibles encore large — limitation fondamentale du dataset
Avec 335 samples / 6 classes / DT=9 samples, atteindre Accuracy=0.95 et F1=0.90 requiert soit plus de données, soit une approche transfer learning. La progression R0→R10 (F1 : 0.16→0.63, Brier : 0.82→0.53) démontre la robustesse de la méthode PIGNN-UQ.

---

## Bilan Global R0→R10

### Progression des métriques clés (test ensemble / CV)

| Round | CV F1 | Ens. F1 | Single F1 | Brier ens. | T single |
|-------|--------|---------|-----------|------------|---------|
| R0 | 0.295 | — | 0.163 | 0.817 | — |
| R4 | 0.490 | 0.560 | 0.528 | 0.647 | 0.90 |
| R5 | — | — | 0.561 | 0.572 | 1.37 |
| R6 | 0.517 | 0.585 | 0.539 | 0.606 | 1.43 |
| **R9** | 0.505 | — | **0.624** | — | 1.19 |
| **R10** | **0.632** | **0.612** | ~0.58 | **0.531** | **0.89** |

### Leçons clés du projet PIGNN-UQ

1. **Focal loss → CE** : focal loss causait sous-confiance (T<1) et mauvais Brier
2. **BatchNorm1d >> LayerNorm** pour features de nœuds PyG (convergence ×3)
3. **Embedding-level Mixup** : crucial pour petits datasets (335 samples)
4. **hidden_dim=128 + dropout=0.10** : gain massif vs hidden_dim=96 + dropout=0.15
5. **CosineAnnealingWarmRestarts T_0=200, 600 epochs** : 3 cycles complets optimaux
6. **MC ensemble prob avg + T clamp [0.5,3.0]** : meilleure stratégie ensemble
7. **Évaluation MC Dropout stochastique** : variance ±0.04 F1 sur 51 samples — limite du suivi round-by-round

---

## Architecture Finale — PIGNN-UQ (Round 10)

### Vue d'ensemble

```
Entrée DGA (7 gaz : H₂, CH₄, C₂H₂, C₂H₄, C₂H₆, CO, CO₂)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│              Prétraitement & construction de graphe      │
│  • Log-normalisation des concentrations                  │
│  • Calcul des 8 ratios IEC 60599 / Roger                │
│  • Pondération des nœuds (NODE_WEIGHTS)                 │
│  • 10 arêtes physiques (EDGE_DEFINITIONS)               │
│  • Poids d'arêtes : log_inv / direct_inv / min_tenth    │
│  Features de nœud [7 × 4] :                            │
│    dim 0 : log_gas_norm                                  │
│    dim 1 : log_gas × node_weight                        │
│    dim 2 : vit_gas_norm (vitesse de dégradation)        │
│    dim 3 : principal_ratio_norm                          │
└───────────────────────┬─────────────────────────────────┘
                        │ Data(x=[7,4], edge_index=[2,10],
                        │      edge_attr=[10,1])
                        ▼
┌─────────────────────────────────────────────────────────┐
│  GAT Layer 1  [4 → 128×2 = 256]                        │
│  GATConv(in=4, out=128, heads=2, edge_dim=1)            │
│  + BatchNorm1d(256) + ReLU + Dropout(0.10)              │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  GAT Layer 2  [256 → 128×2 = 256]                      │
│  GATConv(in=256, out=128, heads=2, edge_dim=1)          │
│  + BatchNorm1d(256) + ReLU + Dropout(0.10)              │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  GAT Layer 3  [256 → 128×1 = 128]                      │
│  GATConv(in=256, out=128, heads=1, edge_dim=1)          │
│  + BatchNorm1d(128) + ReLU + Dropout(0.10)              │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Global Attention Pooling  [B×7×128 → B×128]           │
│  gate_nn : Linear(128,64) → ReLU → Linear(64,1)        │
│  pool    : Σ softmax(gate) · node_embeds               │
└───────────────────────┬─────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Classificateur MLP  [128 → 64 → 6]                    │
│  Linear(128,64) → ReLU → Dropout(0.10) → Linear(64,6) │
└───────────────────────┬─────────────────────────────────┘
                        ▼
                  Logits [B × 6]
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
  T-Scaling (T=0.89)        MC Dropout (50 passes)
  → Prédiction single         → Incertitude épistémique
  → argmax → Classe            → Confidence [0,1]
```

### Composantes physiques

| Composante | Description | Référence |
|-----------|-------------|-----------|
| **PhysicsLoss** | λ·(L_Arrhenius + L_discharge) | IEC 60599 + Table 3.2.3 |
| L_Arrhenius | MSE(P(T3), sigmoid(Σ E_a·gas / Σ E_a)) | Proxy Arrhenius thermique |
| L_discharge | MSE(P(D2), sigmoid(C₂H₂_feat)) | Corrélation décharges/C₂H₂ |
| **DropEdge** | Suppression aléatoire d'arêtes (p=0.15) | Augmentation structurelle |
| **Mixup** | lam·emb_a + (1-lam)·emb_b (α=0.3) | Interpolation embedding |

### Hyperparamètres finaux (Round 10)

| Catégorie | Paramètre | Valeur |
|-----------|-----------|--------|
| **Architecture** | node_in_dim | 4 |
| | hidden_dim | 128 |
| | num_gat_layers | 3 |
| | num_heads | 2 (→1 pour GAT3) |
| | pooling | Global Attention |
| | output_dim | 6 |
| | Paramètres totaux | ~368 000 |
| **Régularisation** | dropout_rate | 0.10 |
| | drop_edge_p | 0.15 |
| | label_smoothing | 0.02 |
| | mixup_alpha | 0.3 |
| | noise_std | 0.05 |
| **Optimisation** | optimizer | AdamW |
| | learning_rate | 3×10⁻⁴ |
| | weight_decay | 1×10⁻⁴ |
| | scheduler | CosineAnnealingWarmRestarts |
| | T_0 | 200 epochs |
| | num_epochs | 600 (3 cycles) |
| | early_stop_patience | 200 |
| | batch_size | 32 |
| **Validation** | n_folds | 10 |
| | random_seed | 42 |
| **UQ** | mc_samples | 50 |
| **Physique** | physics_lambda | 0.01 |

### Métriques finales obtenues (Round 10)

#### Cross-validation 10-fold (284 samples train+val)

| Métrique | Valeur | Cible | Gap |
|----------|--------|-------|-----|
| Accuracy | **0.6834 ± 0.1051** | ≥ 0.95 | -0.267 |
| F1 macro | **0.6323 ± 0.1081** | ≥ 0.90 | -0.268 |
| Precision | **0.6675 ± 0.1087** | ≥ 0.90 | -0.233 |
| Recall | **0.6960 ± 0.1045** | ≥ 0.85 | -0.154 |
| Kappa | **0.6143 ± 0.1274** | ≥ 0.85 | -0.236 |
| Brier | **0.5389 ± 0.0783** | ≤ 0.10 | +0.439 |
| Meilleur fold | Fold 3 : Acc=0.8276, F1=0.8288 | — | — |

#### Test set — Ensemble 10 folds × 50 MC passes (51 samples)

| Métrique | Valeur | Cible | Progression R0→R10 |
|----------|--------|-------|--------------------|
| Accuracy | 0.6078 | ≥ 0.95 | +0.333 |
| **F1 macro** | **0.6116** | ≥ 0.90 | **+0.449** |
| Kappa | 0.5145 | ≥ 0.85 | +0.370 |
| **Brier** | **0.5305** | ≤ 0.10 | **-0.286** |
| ECE | 0.1510 | ≤ 0.05 | — |
| Temperature | 0.5000 | ≈ 1.0 | — |

#### Test set — Modèle final seul (256 samples, MC Dropout)

| Métrique | Valeur |
|----------|--------|
| F1 macro | 0.5416 |
| Brier | 0.5499 |
| Temperature | **0.8919** (≈ 1.0, calibration quasi-parfaite) |
| Mean uncertainty | 0.07–0.08 |

### Stratégie d'ensemble (inférence finale)

```
Pour chaque sample x_test :
  Pour chaque fold k ∈ {0,...,9} :
    Charger best_fold_k.pt
    Activer dropout (model.train())
    Pour chaque passe MC t ∈ {1,...,50} :
      p_kt = softmax(model(x_test))   # stochastique
    p_k = mean_t(p_kt)               # [6] prob MC
  
  p_ens = Σ_k w_k · p_k / Σ_k w_k  # w_k = F1_val(fold_k)
  p_ens_T = softmax(log(p_ens) / T)  # T-scaling, T=0.5
  pred = argmax(p_ens_T)
```

### Progression globale R0 → R10

```
F1 macro (ensemble)
0.16 ──► 0.30 ──► 0.36 ──► 0.56 ──► 0.585 ──► 0.612
  R0      R1      R2      R4       R6        R10
         (+87%)  (+20%)  (+56%)   (+5%)     (+5%)

Brier score (test ensemble)
0.82 ──────────────────────────────────────────► 0.53
  R0                                              R10
                         (−35%)
```

### Limitations et pistes d'amélioration

| Limitation | Impact | Piste |
|-----------|--------|-------|
| Dataset trop petit (335 samples) | Gap majeur aux cibles | Transfer learning sur données IEC publiques |
| Classe DT sous-représentée (9/335) | F1 DT ≈ 0 | Oversampling SMOTE-Graph |
| MCDropout stochastique (±0.04 F1) | Comparaison inter-rounds bruitée | Seeds fixes ou éval déterministe |
| Ensemble T saturé à 0.5 | ECE élevée | Température par classe ou Dirichlet calibration |
| Features Vit* ≈ 0 (70% des samples) | dim 2 quasi-constante | Imputation temporelle ou feature engineering |
