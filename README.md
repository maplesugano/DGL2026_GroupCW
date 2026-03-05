# DGL2026 Group Coursework — Brain Graph Super-Resolution

Predict **high-resolution (HR) brain connectivity graphs** from paired **low-resolution (LR)** inputs using graph neural networks. The challenge follows a Kaggle-style evaluation: models are trained on `lr_train` / `hr_train` and must generate predictions for `lr_test`.

---

## Table of Contents

1. [Problem Overview](#problem-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Models](#models)
5. [Pipeline](#pipeline)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Setup](#setup)
8. [Running the Notebooks](#running-the-notebooks)
9. [Outputs](#outputs)

---

## Problem Overview

Each brain connectivity graph is stored as a flattened upper-triangular edge-weight vector. The goal is to super-resolve a 160-node LR graph (12 720 edges) into a 268-node HR graph (35 778 edges) per subject.

---

## Dataset

| File | Shape | Description |
|---|---|---|
| `data/lr_train.csv` | 167 × 12 720 | LR edge vectors, training subjects |
| `data/hr_train.csv` | 167 × 35 778 | HR edge vectors, training subjects |
| `data/lr_test.csv` | 112 × 12 720 | LR edge vectors, test subjects |

Preprocessing steps applied to all splits:
- Replace NaN and negative values with 0
- Remove outlier training subjects (HR mean > mean + 3σ)
- Anti-vectorise LR vectors → adjacency matrices of shape 160 × 160

---

## Project Structure

```
.
├── baseline.ipynb          # SGC baseline model + 3-fold CV
├── improvement.ipynb       # Improved SGC + LowRank generator + ensembling
├── DGL_Dataset_EDA.ipynb   # Exploratory data analysis
├── requirements.txt
├── data/
│   ├── hr_train.csv
│   ├── lr_train.csv
│   └── lr_test.csv
├── figs/                   # Saved bar-charts and plots
├── outputs/                # Per-fold prediction CSVs and submission files
├── report_artifacts/
│   └── pipeline_flowchart.mmd
└── utils/
    ├── MatrixVectorizer.py  # Vectorise / anti-vectorise adjacency matrices
    ├── plot_chart.py        # Bar-chart helpers
    ├── reproducibility.py   # Seed + device setup
    └── training.py          # compute_metrics (MAE, PCC, JSD, graph centralities)
```

---

## Models

### Baseline — SGC (`baseline.ipynb`)

Simple Graph Convolution (SGC) used as a direct regression head:

1. Normalise the LR adjacency matrix: $\hat{A} = D^{-1/2}(A+I)D^{-1/2}$
2. K-step propagation: $X' = \hat{A}^K X$ &nbsp;(K = 2)
3. Linear node projection → flatten → MLP → 35 778-dim HR vector

Key hyperparameters: `HIDDEN_DIM=128`, `MLP_DIM=256`, `EPOCHS=200`, `LR=1e-3`, `BATCH_SIZE=16`.

---

### Improved SGC (`improvement.ipynb` — Phase 1)

Enhancements on top of the baseline:

| Change | Detail |
|---|---|
| Sigmoid output | Constrains predictions to [0, 1] |
| Dropout (p = 0.3) | Applied inside MLP |
| AdamW | Weight decay = 1e-3 |
| Early stopping | Patience = 20 epochs, best-weight restore |
| Flexible loss | L1 (MAE) or Weighted-MAE (`ALPHA=5.0`) |

---

### Low-Rank HR Generator (`improvement.ipynb` — Phase 2)

An alternative decoder that predicts the HR adjacency matrix via low-rank factorisation:

1. Encode LR adjacency with SGC propagation + mean-pool → graph vector $g$
2. Decode $g$ → node embeddings $Z \in \mathbb{R}^{B \times 268 \times d}$
3. $\hat{A} = \sigma(Z Z^\top)$, symmetric, zero diagonal, upper-triangular vectorised

Switch with `MODEL_TYPE = "sgc"` or `"lowrank"` and `LOWRANK_DIM = 32`.

---

### Ensembling (`improvement.ipynb` — Phase 3)

- **Fold averaging**: mean predictions across 3 CV folds (seed = 42)
- **Seed averaging**: optionally repeat for seeds `[42, 123, 999]` and average (`RUN_SEED_AVG = True`)
- Final predictions are clipped to [0, 1] and serialised to a flat submission CSV

---

## Pipeline

```mermaid
flowchart TB
  classDef inputClass  fill:#ff6b6b,stroke:#c0392b,color:#fff
  classDef prepClass   fill:#6ab04c,stroke:#4a7c30,color:#fff
  classDef cvClass     fill:#686de0,stroke:#4a4fbf,color:#fff
  classDef modelClass  fill:#f6c667,stroke:#c9940a,color:#333
  classDef reportClass fill:#63cdda,stroke:#2e9ab0,color:#333
  classDef testClass   fill:#4cd137,stroke:#27ae20,color:#333
  classDef outputClass fill:#e84393,stroke:#b0135f,color:#fff
  classDef loopClass   fill:#fd9644,stroke:#c9650a,color:#333

  A[("Input CSVs<br/>lr_train: (167,12720)<br/>hr_train: (167,35778)<br/>lr_test: (112,12720)")]:::inputClass

  subgraph P["Pre-processing"]
    P1["Load vectors<br/>pandas -> numpy float32"]:::prepClass
    P2["Sanitise: NaN->0, negative->0"]:::prepClass
    P3["Outlier removal<br/>(HR mean > mean + 3σ)"]:::prepClass
    P4["Anti-vectorise LR<br/>xLR -> A_LR (160×160)"]:::prepClass
    P5["Targets: keep HR as vector<br/>xHR (35778)"]:::prepClass
  end

  A --> P1 --> P2 --> P3 --> P4 --> P5

  subgraph CV["3-Fold Cross-Validation"]
    S2{"For each fold"}:::loopClass
    S3["Split subjects<br/>train_idx / val_idx"]:::cvClass
    S4["DataLoaders<br/>batch=16"]:::cvClass
    S5["Build model"]:::cvClass
    S6["Optimiser: AdamW<br/>lr=1e-3, wd=1e-3"]:::cvClass
    S7["Loss: MAE"]:::cvClass
    S8["Train epochs <= 200"]:::cvClass
    S9["Early stopping<br/>patience=20<br/>restore best weights"]:::cvClass
    S10["Val inference + postprocess<br/>clip to [0,1]"]:::cvClass
    S11["Save fold preds<br/>predictions_fold_k.csv"]:::cvClass
    S12["Compute metrics on fold"]:::cvClass
  end

  P5 --> S2
  S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8 --> S9 --> S10 --> S11 --> S12 --> S2

  subgraph M["Model options"]
    M1["SGC Baseline<br/>A_norm^K propagation<br/>+ node projection<br/>+ MLP head<br/>(dropout=0.3)"]:::modelClass
    M2["LowRank HR Generator<br/>Encode LR via SGC<br/>Decode g->Z (268×d)<br/>A_hat = Z·Zᵀ (sym, diag=0)<br/>extract upper-tri vec"]:::modelClass
  end
  S5 -.selects.-> M1
  S5 -.selects.-> M2

  subgraph R["Cross-fold reporting"]
    R1["Aggregate fold metrics<br/>mean ± std"]:::reportClass
    R2["Bar plots across folds<br/>(all measures)"]:::reportClass
  end

  S12 --> R1 --> R2

  subgraph T["Inference + ensembling"]
    T1["Anti-vectorise lr_test<br/>xLR -> A_LR (160×160)"]:::testClass
    T2["Run CV training again (seed=42)<br/>predict test for each fold"]:::testClass
    T3["Fold-avg test preds<br/>mean over 3 folds"]:::testClass
    T4["Optional seed-avg<br/>repeat for seeds [42,123,999]<br/>mean over seeds"]:::testClass
    T5["Postprocess clip to [0,1]"]:::testClass
    T6["Serialise predictions<br/>112×35778 -> flatten 4,007,136"]:::testClass
    T7["Write submission CSV<br/>Predicted=flattened"]:::testClass
  end

  A --> T1 --> T2 --> T3 --> T4 --> T5 --> T6 --> T7

  R2 --> O[("Artifacts<br/>figs/*.png<br/>outputs/predictions_fold_*.csv<br/>outputs/submission_*.csv")]:::outputClass
  T7 --> O
```

---

## Evaluation Metrics

All metrics are computed per-subject via `utils/training.compute_metrics` and averaged:

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error on raw edge weights |
| **PCC** | Pearson Correlation Coefficient |
| **JSD** | Jensen–Shannon Divergence |
| **MAE(BC)** | MAE of betweenness centrality |
| **MAE(EC)** | MAE of eigenvector centrality |
| **MAE(PC)** | MAE of PageRank centrality |
| **MAE(CC)** | MAE of closeness centrality |
| **MAE(DC)** | MAE of degree centrality |

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric accelerate -U
```

> GPU (CUDA 11.8) is recommended but the code falls back to CPU automatically.

---

## Running the Notebooks

| Notebook | Purpose |
|---|---|
| `DGL_Dataset_EDA.ipynb` | Exploratory analysis of LR/HR distributions |
| `baseline.ipynb` | Train & evaluate the SGC baseline |
| `improvement.ipynb` | Train improved SGC / LowRank model, run ensembling, generate submission |

Run all cells top-to-bottom. Intermediate predictions are saved to `outputs/` after each fold.

---

## Outputs

| Path | Contents |
|---|---|
| `outputs/predictions_fold_k.csv` | Validation predictions for fold k |
| `outputs/submission_*.csv` | Final Kaggle submission (`ID`, `Predicted`) |
| `figs/*.png` | Bar charts of per-fold metrics |

