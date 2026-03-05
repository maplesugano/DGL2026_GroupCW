"""Graph metric computation utilities."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_subject_metrics(pred_vec: np.ndarray, gt_vec: np.ndarray) -> dict:
    """Compute all graph metrics for a single subject (designed for parallel execution)."""
    import networkx as nx
    import numpy as np
    from sklearn.metrics import mean_absolute_error
    from utils.MatrixVectorizer import MatrixVectorizer

    pred_mat = MatrixVectorizer.anti_vectorize(pred_vec, 268)
    gt_mat = MatrixVectorizer.anti_vectorize(gt_vec, 268)

    pred_graph = nx.from_numpy_array(pred_mat, edge_attr="weight")
    gt_graph = nx.from_numpy_array(gt_mat, edge_attr="weight")

    pred_bc = nx.betweenness_centrality(pred_graph, weight="weight")
    pred_ec = nx.eigenvector_centrality(pred_graph, weight="weight", max_iter=1000)
    pred_pc = nx.pagerank(pred_graph, weight="weight")
    pred_cc = nx.closeness_centrality(pred_graph)
    pred_dc = nx.degree_centrality(pred_graph)

    gt_bc = nx.betweenness_centrality(gt_graph, weight="weight")
    gt_ec = nx.eigenvector_centrality(gt_graph, weight="weight", max_iter=1000)
    gt_pc = nx.pagerank(gt_graph, weight="weight")
    gt_cc = nx.closeness_centrality(gt_graph)
    gt_dc = nx.degree_centrality(gt_graph)

    return {
        "mae_bc": mean_absolute_error(list(pred_bc.values()), list(gt_bc.values())),
        "mae_ec": mean_absolute_error(list(pred_ec.values()), list(gt_ec.values())),
        "mae_pc": mean_absolute_error(list(pred_pc.values()), list(gt_pc.values())),
        "mae_cc": mean_absolute_error(list(pred_cc.values()), list(gt_cc.values())),
        "mae_dc": mean_absolute_error(list(pred_dc.values()), list(gt_dc.values())),
        "pred_1d": MatrixVectorizer.vectorize(pred_mat),
        "gt_1d": MatrixVectorizer.vectorize(gt_mat),
    }


def compute_metrics(
    pred_vecs: np.ndarray,
    gt_vecs: np.ndarray,
    sanitize_fn=None,
) -> dict:
    """Compute MAE, PCC, JSD and graph-topology MAEs across all subjects.

    Parameters
    ----------
    pred_vecs, gt_vecs:
        Arrays of shape ``(N, edge_dim)``.
    sanitize_fn:
        Optional callable applied to both arrays before evaluation
        (e.g. ``sanitize_vectors`` from the notebook).
    """
    if sanitize_fn is not None:
        pred_vecs = sanitize_fn(pred_vecs)
        gt_vecs = sanitize_fn(gt_vecs)

    results = Parallel(n_jobs=-1)(
        delayed(_compute_subject_metrics)(pred_vecs[i], gt_vecs[i])
        for i in range(pred_vecs.shape[0])
    )

    mae_bc = np.mean([r["mae_bc"] for r in results])
    mae_ec = np.mean([r["mae_ec"] for r in results])
    mae_pc = np.mean([r["mae_pc"] for r in results])
    mae_cc = np.mean([r["mae_cc"] for r in results])
    mae_dc = np.mean([r["mae_dc"] for r in results])

    pred_1d = np.concatenate([r["pred_1d"] for r in results])
    gt_1d   = np.concatenate([r["gt_1d"]   for r in results])

    mae = mean_absolute_error(gt_1d, pred_1d)
    pcc = pearsonr(pred_1d, gt_1d)[0]
    jsd = jensenshannon(pred_1d, gt_1d)

    return {
        "MAE": mae,
        "PCC": pcc,
        "JSD": jsd,
        "MAE(PC)": float(mae_pc),
        "MAE(EC)": float(mae_ec),
        "MAE(BC)": float(mae_bc),
        "MAE(CC)": float(mae_cc),
        "MAE(DC)": float(mae_dc),
    }
