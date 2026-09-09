"""Loading and basic geometry for the ShapeNet-Car pressure dataset.

Two mesh types live under data/: 611 triangulated ("Class A", the official
train/test split, 3586 verts) and 187 unsplit quad meshes ("Class B", 3682 verts,
not used here). Class A's pressure files need a 96-entry offset fix -- see
load_pressure() and the README for how that was found.
"""
from __future__ import annotations

import os

import numpy as np
from plyfile import PlyData

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "processed-car-pressure-data")
DATA_DIR = os.path.join(DATASET_ROOT, "data")

CLASS_A_PRESSURE_OFFSET = 96


def read_split(name: str) -> list[str]:
    """Read train.txt or test.txt -> list of zero-padded 3-digit ids."""
    path = os.path.join(DATASET_ROOT, name)
    with open(path) as f:
        return f.read().strip().split(",")


def make_val_split(train_ids: list[str], n_val: int = 50, seed: int = 0) -> tuple[list[str], list[str]]:
    """Carve a validation set out of the official train ids (no val split is provided)."""
    rng = np.random.default_rng(seed)
    ids = list(train_ids)
    rng.shuffle(ids)
    return ids[n_val:], ids[:n_val]


def load_mesh(sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load mesh_<id>.ply -> (verts (V,3) float32, faces (F,3) int64, triangulated).

    Class A files are pure triangles already. Class B files are pure quads and
    are triangulated by splitting each quad into two triangles (0,1,2) and (0,2,3).
    """
    path = os.path.join(DATA_DIR, f"mesh_{sample_id}.ply")
    ply = PlyData.read(path)
    v = ply["vertex"]
    verts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    raw_faces = ply["face"]["vertex_indices"]
    face_sizes = {len(f) for f in raw_faces}
    if face_sizes == {3}:
        faces = np.stack(raw_faces).astype(np.int64)
    elif face_sizes == {4}:
        quads = np.stack(raw_faces).astype(np.int64)
        faces = np.concatenate(
            [quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=0
        )
    else:
        raise ValueError(f"{sample_id}: mixed/unsupported face sizes {face_sizes}")

    return verts, faces


def load_pressure(sample_id: str, n_verts: int) -> np.ndarray:
    """Load press_<id>.npy and align it to the mesh's vertex count.

    n_verts == 3586 -> Class A, apply the verified press[96:] offset.
    n_verts == 3682 -> Class B, use as-is.
    """
    path = os.path.join(DATA_DIR, f"press_{sample_id}.npy")
    press = np.load(path).astype(np.float32)

    if n_verts == 3586:
        press = press[CLASS_A_PRESSURE_OFFSET:]
    elif n_verts == 3682:
        pass
    else:
        raise ValueError(f"{sample_id}: unexpected vertex count {n_verts}")

    if press.shape[0] != n_verts:
        raise AssertionError(
            f"{sample_id}: pressure length {press.shape[0]} != vertex count {n_verts}"
        )
    return press


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted face-normal accumulation per vertex (no normals are stored)."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)  # magnitude == 2 * area, direction weights by area

    normals = np.zeros_like(verts)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (normals / norms).astype(np.float32)


def face_areas_and_normals(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (face_areas (F,), face_unit_normals (F,3))."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    safe = areas.copy()
    safe[safe == 0] = 1.0
    unit_normals = cross / (2 * safe[:, None])
    return areas.astype(np.float32), unit_normals.astype(np.float32)


def projected_frontal_area(verts: np.ndarray, faces: np.ndarray, flow_axis: int = 2) -> float:
    """Silhouette area seen along flow_axis, as 0.5 * sum(|n_axis| * area).

    Exact for a convex closed surface (every line along the axis crosses it twice),
    and a slight over-estimate for the wheel arches and underbody. Preferred over a
    bounding-box cross-section, which is set by four extreme vertices -- barely
    differentiable, and trivially inflated by an optimizer to shrink anything it
    divides.
    """
    areas, normals = face_areas_and_normals(verts, faces)
    return float(0.5 * (np.abs(normals[:, flow_axis]) * areas).sum())


def enclosed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Volume enclosed by a closed mesh, via the divergence theorem.

    Positive only when face windings give outward normals, so this doubles as the
    orientation check behind CD_PHYSICAL_SIGN (see the README).
    """
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(((v0 + v1 + v2) * np.cross(v1 - v0, v2 - v0)).sum() / 18.0)


def pressure_drag_force(
    verts: np.ndarray, faces: np.ndarray, pressure: np.ndarray, flow_axis: int = 2
) -> float:
    """Un-normalized pressure-drag integral -- the numerator of the Cd proxy.

    This is the quantity that shape optimization should actually reduce; dividing by
    a shape-dependent area lets a larger car score better without being less draggy.
    """
    areas, normals = face_areas_and_normals(verts, faces)
    face_pressure = pressure[faces].mean(axis=1)
    return float((face_pressure * normals[:, flow_axis] * areas).sum())


def compute_drag_proxy(
    verts: np.ndarray,
    faces: np.ndarray,
    pressure: np.ndarray,
    flow_axis: int = 2,
    reference_area: float | None = None,
) -> float:
    """Pressure-drag integral, used as a Cd stand-in since the dataset has no real one.

    Pressure drag only (no viscous term, no velocity/density normalization) -- see
    the README for what this leaves out and for the sign correction applied later
    in src/shape_opt.py.

    Pass `reference_area` to normalize by a fixed area instead of the shape's own.
    Shape optimization must do this: with the area recomputed each step, growing the
    car is a valid way to lower the ratio, so the optimizer pursues it instead of
    reducing drag.
    """
    raw_drag = pressure_drag_force(verts, faces, pressure, flow_axis)
    if reference_area is None:
        reference_area = projected_frontal_area(verts, faces, flow_axis)
    return float(raw_drag / max(reference_area, 1e-6))


def mesh_geometry_summary(verts: np.ndarray, faces: np.ndarray, flow_axis: int = 2) -> dict:
    """Total surface area and frontal (cross-sectional) area -- both scale with length^2,
    so they're reported in the mesh's own coordinate units (comparable before/after a
    shape-optimization run in the same normalized space, not to real-world units).
    """
    areas, _ = face_areas_and_normals(verts, faces)
    extent = verts.max(axis=0) - verts.min(axis=0)
    return {
        "surface_area": float(areas.sum()),
        "frontal_area": projected_frontal_area(verts, faces, flow_axis),
        "bbox_frontal_area": float(extent[(flow_axis + 1) % 3] * extent[(flow_axis + 2) % 3]),
        "volume": enclosed_volume(verts, faces),
        "length": float(extent[flow_axis]),
    }


def normalize_positions(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Center + scale a single mesh into a unit bounding box. Returns (verts_norm, center, scale)."""
    bbox_min, bbox_max = verts.min(axis=0), verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    scale = float(np.max(bbox_max - bbox_min) / 2.0)
    scale = scale if scale > 0 else 1.0
    return (verts - center) / scale, center, scale
