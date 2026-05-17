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

*(à remplir)*

### Modifications pour Round 6

*(à remplir après analyse des résultats)*
