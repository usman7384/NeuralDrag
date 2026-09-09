"""Evaluation metrics for the surrogate and the shape optimization.

Everything the README's Results section quotes is computed here, so each claim
traces to a function rather than to a one-off script. The baselines matter as much
as the model's own score: all these cars are broadly car-shaped, so simply reciting
the training set's average pressure field is a surprisingly strong predictor, and a
model that failed to beat it would be doing nothing useful while still posting a
respectable-looking error.
"""
from __future__ import annotations

import numpy as np
import torch

from src.data import compute_drag_proxy, load_mesh


def predict_test_set(model, pyg_data, stats: dict, device: str = "cpu"):
    """Run the model over a list of PyG samples -> (pressure (N,V), cd (N,)) in original units."""
    model.eval()
    pressure, cd = [], []
    for s in pyg_data:
        with torch.no_grad():
            p, c = model(
                s.x.to(device),
                s.edge_index.to(device),
                torch.zeros(s.x.shape[0], dtype=torch.long, device=device),
            )
        pressure.append(p.cpu().numpy() * stats["pressure_std"] + stats["pressure_mean"])
        cd.append(float(c.cpu()) * stats["cd_std"] + stats["cd_mean"])
    return np.stack(pressure), np.array(cd)


def regression_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """MSE, RMSE and R^2. RMSE is in the target's own units, so it is the one to
    compare an optimization's claimed improvement against."""
    mse = float(((pred - true) ** 2).mean())
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(1 - mse / true.var()) if true.var() > 0 else float("nan"),
    }


def pressure_baselines(train_pressure: np.ndarray, test_pressure: np.ndarray) -> dict:
    """Two geometry-blind predictors, scored on the test set.

    `global_mean` predicts one scalar everywhere. `template_field` predicts the
    training set's per-vertex mean pressure, ignoring which car it is looking at --
    the meaningful bar, since it captures everything true of cars in general.
    """
    return {
        "global_mean": float(((test_pressure - train_pressure.mean()) ** 2).mean()),
        "template_field": float(((test_pressure - train_pressure.mean(axis=0)) ** 2).mean()),
    }


def analytic_cd_from_pressure(samples: list[dict], pressure_pred: np.ndarray) -> np.ndarray:
    """Re-derive Cd from predicted pressure via the same formula used to build the
    labels -- the cross-check described in the README's 'How to judge' section."""
    return np.array(
        [
            compute_drag_proxy(s["pos"], s["faces"], pressure_pred[i])
            for i, s in enumerate(samples)
        ]
    )


def deformation_roughness(displacement: np.ndarray, edge_index: np.ndarray) -> float:
    """Fraction of a displacement field that is high-frequency: ||L(d)|| / ||d||,
    where L is the graph Laplacian over mesh edges.

    ~0 means a coherent, smooth reshaping; ~1 means per-vertex noise that happens to
    average out. Distinguishes a genuine shape change from surface jitter, which a
    displacement magnitude alone cannot do.
    """
    src, dst = edge_index
    neighbor_sum = np.zeros_like(displacement)
    counts = np.zeros(len(displacement))
    np.add.at(neighbor_sum, src, displacement[dst])
    np.add.at(counts, src, 1)
    laplacian = displacement - neighbor_sum / np.maximum(counts, 1)[:, None]
    magnitude = np.linalg.norm(displacement, axis=1).mean()
    return float(np.linalg.norm(laplacian, axis=1).mean() / max(magnitude, 1e-12))


def noise_floor_ratio(multi_results: list[dict], cd_rmse: float) -> dict:
    """Compare each optimization run's claimed Cd change against the surrogate's own
    test RMSE. A ratio below 1 means the claimed improvement is smaller than the
    model's typical error on that quantity, i.e. indistinguishable from no change.
    """
    changes = np.array(
        [abs(r["after"]["cd_head_raw"] - r["before"]["cd_head_raw"]) for r in multi_results]
    )
    return {
        "mean_abs_cd_change": float(changes.mean()),
        "cd_rmse": cd_rmse,
        "ratio": float(changes.mean() / cd_rmse),
        "n_exceeding_rmse": int((changes > cd_rmse).sum()),
        "n_total": len(changes),
    }
