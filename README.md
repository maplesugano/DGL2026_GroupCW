# DGL2026 Group Coursework — Brain Graph Super-Resolution

Predict **high-resolution (HR) brain connectivity graphs** from paired **low-resolution (LR)** inputs using graph neural networks. The challenge follows a Kaggle-style evaluation: models are trained on `lr_train` / `hr_train` and must generate predictions for `lr_test`.

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

## Pipeline
![Pipeline Flowchart](report_artifacts/pipeline_flowchart.png)

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

## Outputs

| Path | Contents |
|---|---|
| `outputs/predictions_fold_k.csv` | Validation predictions for fold k |
| `outputs/submission_*.csv` | Final Kaggle submission (`ID`, `Predicted`) |
| `figs/*.png` | Bar charts of per-fold metrics |

