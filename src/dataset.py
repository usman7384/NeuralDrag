"""Builds PyG Data objects and a cached in-memory dataset for the car meshes."""
from __future__ import annotations

import os

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from src.data import (
    compute_drag_proxy,
    compute_vertex_normals,
    load_mesh,
    load_pressure,
    normalize_positions,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "cache")


def faces_to_edge_index(faces: np.ndarray) -> np.ndarray:
    """Undirected mesh edges (both directions, deduplicated) from triangle faces."""
    e = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    e = np.concatenate([e, e[:, ::-1]], axis=0)
    e = np.unique(e, axis=0)
    return e.T  # (2, E)


def build_sample(sample_id: str) -> dict:
    """Load + featurize a single car -> plain dict of numpy arrays (pre-normalization)."""
    verts, faces = load_mesh(sample_id)
    pressure = load_pressure(sample_id, verts.shape[0])
    normals = compute_vertex_normals(verts, faces)
    cd_proxy = compute_drag_proxy(verts, faces, pressure)
    verts_norm, center, scale = normalize_positions(verts)
    edge_index = faces_to_edge_index(faces)

    return {
        "id": sample_id,
        "pos": verts_norm.astype(np.float32),
        "normal": normals.astype(np.float32),
        "edge_index": edge_index.astype(np.int64),
        "faces": faces.astype(np.int64),
        "pressure": pressure.astype(np.float32),
        "cd": np.float32(cd_proxy),
        "center": center.astype(np.float32),
        "scale": np.float32(scale),
    }


def build_all(ids: list[str], cache_name: str) -> list[dict]:
    """Build (or load from cache) a list of raw sample dicts for the given ids."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)

    samples = [build_sample(sid) for sid in tqdm(ids, desc=f"building {cache_name}")]
    torch.save(samples, cache_path)
    return samples


class MeshData(Data):
    """A Data subclass that knows how to batch `face` (like edge_index -- indices
    into this graph's nodes, so PyG must offset them by the running node count when
    stacking graphs into a batch) and `face_batch` (a per-face graph-id counter, like
    the node-level `batch` vector, so per-graph quantities -- e.g. the drag integral --
    can be computed with scatter ops across a batch of differently-sized meshes).
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "face":
            return self.num_nodes
        if key == "face_batch":
            return 1
        return super().__inc__(key, value, *args, **kwargs)


def to_pyg_data(sample: dict, pressure_mean: float, pressure_std: float, cd_mean: float, cd_std: float) -> MeshData:
    """Standardize targets (using precomputed train stats) and wrap as a PyG Data object."""
    x = np.concatenate([sample["pos"], sample["normal"]], axis=1)
    y_pressure = (sample["pressure"] - pressure_mean) / pressure_std
    y_cd = (sample["cd"] - cd_mean) / cd_std
    faces = sample["faces"]

    return MeshData(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(sample["edge_index"]),
        face=torch.from_numpy(faces.T),  # PyG convention: (3, F), like edge_index is (2, E)
        face_batch=torch.zeros(faces.shape[0], dtype=torch.long),
        y_pressure=torch.from_numpy(y_pressure),
        y_cd=torch.tensor([y_cd], dtype=torch.float32),
        sample_id=sample["id"],
    )


def compute_train_stats(train_samples: list[dict]) -> dict:
    all_pressure = np.concatenate([s["pressure"] for s in train_samples])
    all_cd = np.array([s["cd"] for s in train_samples])
    return {
        "pressure_mean": float(all_pressure.mean()),
        "pressure_std": float(all_pressure.std()),
        "cd_mean": float(all_cd.mean()),
        "cd_std": float(all_cd.std()),
    }
