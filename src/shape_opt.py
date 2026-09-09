"""Differentiable shape optimization over an FFD cage.

Two things distinguish this from descending on raw vertex positions:

1. The optimizable variables are FFD control points (see src/ffd.py), a few hundred
   degrees of freedom instead of 3*V, and every deformation they can express is a
   smooth polynomial -- so the optimizer cannot reach the noisy, adversarial meshes
   that free vertices allow.
2. The objective is predicted drag *force* (Cd times projected frontal area), not Cd.
   Minimizing Cd alone rewards growing the car, since frontal area sits in Cd's
   denominator; drag force is what actually opposes motion.
"""
from __future__ import annotations

import numpy as np
import torch

from src.data import (
    compute_drag_proxy,
    mesh_geometry_summary,
    pressure_drag_force,
    projected_frontal_area,
)
from src.ffd import FFDLattice

# compute_drag_proxy() comes out consistently negative for reasons that have nothing
# to do with physics (mesh normal winding + the chosen flow axis). This flips it to
# the sign that actually matches real drag -- see the README for how it was derived.
CD_PHYSICAL_SIGN = -1.0


def differentiable_vertex_normals(pos: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals as a function of `pos`, differentiable end-to-end.

    faces: (F, 3) long tensor of vertex indices (constant topology during optimization).
    """
    v0, v1, v2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # magnitude encodes area

    normals = torch.zeros_like(pos)
    normals.index_add_(0, faces[:, 0], face_normals)
    normals.index_add_(0, faces[:, 1], face_normals)
    normals.index_add_(0, faces[:, 2], face_normals)

    return torch.nn.functional.normalize(normals, dim=1, eps=1e-8)


def differentiable_projected_area(
    pos: torch.Tensor, faces: torch.Tensor, flow_axis: int = 2
) -> torch.Tensor:
    """Differentiable counterpart of src.data.projected_frontal_area.

    |cross[axis]| == 2 * area * |normal[axis]|, so the 0.5*sum(|n|*A) definition
    collapses to a quarter of the summed absolute cross-product component.
    """
    v0, v1, v2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    cross = torch.cross(v1 - v0, v2 - v0, dim=1)
    return 0.25 * cross[:, flow_axis].abs().sum()


def differentiable_enclosed_volume(pos: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Differentiable counterpart of src.data.enclosed_volume."""
    v0, v1, v2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    return ((v0 + v1 + v2) * torch.cross(v1 - v0, v2 - v0, dim=1)).sum() / 18.0


def differentiable_drag_proxy_batch(
    pos: torch.Tensor,
    node_batch: torch.Tensor,
    face: torch.Tensor,
    face_batch: torch.Tensor,
    pressure: torch.Tensor,
    flow_axis: int = 2,
) -> torch.Tensor:
    """Batched, differentiable version of `src.data.compute_drag_proxy` -- one value
    per graph in a PyG batch, using `face_batch` to know which faces belong to which
    graph (see MeshData in src/dataset.py for how that's built during batching).
    Same raw sign convention as compute_drag_proxy (CD_PHYSICAL_SIGN is not applied
    here, so this matches the model's raw cd_pred / the training labels directly).

    `face` follows PyG's convention of shape (3, F), like edge_index is (2, E).
    """
    from torch_scatter import scatter_add

    v0, v1, v2 = pos[face[0]], pos[face[1]], pos[face[2]]
    cross = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_areas = 0.5 * cross.norm(dim=1)
    face_normals = cross / (2 * face_areas.clamp(min=1e-8)).unsqueeze(1)
    face_pressure = pressure[face].mean(dim=0)

    num_graphs = int(node_batch.max().item()) + 1
    raw_drag = scatter_add(
        face_pressure * face_normals[:, flow_axis] * face_areas,
        face_batch, dim=0, dim_size=num_graphs,
    )
    reference_area = 0.25 * scatter_add(
        cross[:, flow_axis].abs(), face_batch, dim=0, dim_size=num_graphs
    )
    return raw_drag / reference_area.clamp(min=1e-6)


def optimize_shape(
    model,
    pos_init: torch.Tensor,
    faces: torch.Tensor,
    edge_index: torch.Tensor,
    n_steps: int = 300,
    lr: float = 2e-3,
    beta_anchor: float = 20.0,
    gamma_area: float = 300.0,
    gamma_volume: float = 300.0,
    max_rel_cd_change: float = 0.10,
    grad_clip_norm: float = 1.0,
    lattice_dims: tuple[int, int, int] = (4, 4, 6),
    cd_mean: float = 0.0,
    cd_std: float = 1.0,
    pressure_mean: float = 0.0,
    pressure_std: float = 1.0,
    flow_axis: int = 2,
    device: str = "cpu",
):
    """Deform a mesh toward lower predicted drag by descending on FFD control points.

    The loss is predicted drag force normalized by its starting value, a trust-region
    term holding the control points near their initial grid, and penalties pinning
    frontal area and enclosed volume to their starting values. No Laplacian or
    gradient-smoothing terms are needed: the FFD basis cannot express the
    high-frequency deformations those existed to suppress.

    The two package penalties are what make the objective mean anything. Drag force
    and Cd are gameable in opposite directions -- a smaller car has less drag force,
    a wider one has lower Cd because frontal area is Cd's denominator -- so without
    them the optimizer just resizes the car instead of restyling it. Holding both
    fixed leaves reshaping as the only way to improve, and makes drag force and Cd
    equivalent objectives.

    Returns a list of dicts with {step, pos, cd_pred, drag_force, pressure} checkpoints.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    pos_init = pos_init.detach().to(device)
    faces = faces.to(device)
    edge_index = edge_index.to(device)
    batch = torch.zeros(pos_init.shape[0], dtype=torch.long, device=device)

    lattice = FFDLattice(pos_init, dims=lattice_dims).to(device)
    ctrl = lattice.control_points.clone().requires_grad_(True)
    ctrl_orig = lattice.control_points.clone()

    optimizer = torch.optim.Adam([ctrl], lr=lr)
    history = []
    force0 = None
    area0 = differentiable_projected_area(pos_init, faces, flow_axis).detach()
    volume0 = differentiable_enclosed_volume(pos_init, faces).detach()

    for step in range(n_steps):
        optimizer.zero_grad()

        pos = lattice.deform(ctrl)
        normals = differentiable_vertex_normals(pos, faces)
        pressure_pred, cd_pred = model(torch.cat([pos, normals], dim=1), edge_index, batch)

        area = differentiable_projected_area(pos, faces, flow_axis)
        cd_physical = CD_PHYSICAL_SIGN * (cd_pred.squeeze() * cd_std + cd_mean)
        drag_force = cd_physical * area
        if force0 is None:
            force0 = float(drag_force.detach().cpu())

        anchor = ((ctrl - ctrl_orig) ** 2).sum(dim=1).mean()
        loss = (
            drag_force / abs(force0)
            + beta_anchor * anchor
            + gamma_area * ((area - area0) / area0) ** 2
            + gamma_volume
            * ((differentiable_enclosed_volume(pos, faces) - volume0) / volume0) ** 2
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_([ctrl], grad_clip_norm)
        optimizer.step()

        force_value = float(drag_force.detach().cpu())
        history.append(
            {
                "step": step,
                "pos": pos.detach().cpu().clone(),
                "cd_pred": float(cd_pred.detach().cpu()),
                "drag_force": force_value,
                "pressure": (pressure_pred.detach().cpu().numpy() * pressure_std + pressure_mean),
            }
        )

        if abs(force_value - force0) > max_rel_cd_change * abs(force0):
            break

    return history


def verify_optimization(
    model,
    pos_before: torch.Tensor,
    pos_after: torch.Tensor,
    faces: torch.Tensor,
    edge_index: torch.Tensor,
    pressure_mean: float,
    pressure_std: float,
    cd_mean: float,
    cd_std: float,
    flow_axis: int = 2,
    device: str = "cpu",
) -> dict:
    """Cross-check the Cd head's claimed improvement against the pressure head.

    Every reported ratio is normalized by the *starting* shape's frontal area, so a
    fatter car cannot post an improvement it did not earn; `frontal_area` is reported
    alongside so any such growth stays visible. `drag_force` is the un-normalized
    integral and is the quantity to judge the run on.

    Note the two heads share an encoder (see src/model.py), so agreement between them
    is a consistency check, not independent evidence. Compare each change against the
    corresponding test-set RMSE before believing it.
    """
    faces_t = faces.to(device)
    edge_index_t = edge_index.to(device)
    faces_np = faces.cpu().numpy()
    ref_area = projected_frontal_area(pos_before.cpu().numpy(), faces_np, flow_axis)

    results = {}
    for name, pos in [("before", pos_before), ("after", pos_after)]:
        pos_d = pos.clone().detach().to(device)
        with torch.no_grad():
            normals = differentiable_vertex_normals(pos_d, faces_t)
            batch = torch.zeros(pos_d.shape[0], dtype=torch.long, device=device)
            pressure_pred, cd_pred = model(
                torch.cat([pos_d, normals], dim=1), edge_index_t, batch
            )

        pos_np = pos_d.cpu().numpy()
        pressure_np = pressure_pred.cpu().numpy() * pressure_std + pressure_mean
        geom = mesh_geometry_summary(pos_np, faces_np, flow_axis)
        cd_head = CD_PHYSICAL_SIGN * (float(cd_pred.cpu().item()) * cd_std + cd_mean)

        results[name] = {
            "cd_head_raw": cd_head,
            "cd_head_drag_force": cd_head * geom["frontal_area"],
            "analytic_cd_from_predicted_pressure": CD_PHYSICAL_SIGN
            * compute_drag_proxy(pos_np, faces_np, pressure_np, flow_axis, ref_area),
            "analytic_drag_force": CD_PHYSICAL_SIGN
            * pressure_drag_force(pos_np, faces_np, pressure_np, flow_axis),
            "pressure_pred_min": float(pressure_np.min()),
            "pressure_pred_max": float(pressure_np.max()),
            **geom,
        }

    disp = (pos_after.to(device) - pos_before.to(device)).norm(dim=1)
    results["displacement"] = {
        "max": float(disp.max().item()),
        "mean": float(disp.mean().item()),
    }
    results["reference_area_used"] = ref_area

    def pct(key):
        b, a = results["before"][key], results["after"][key]
        return (a - b) / abs(b) * 100

    results["cd_head_pct_change"] = pct("cd_head_raw")
    results["analytic_pct_change"] = pct("analytic_cd_from_predicted_pressure")
    results["drag_force_pct_change"] = pct("analytic_drag_force")
    results["frontal_area_pct_change"] = pct("frontal_area")
    results["volume_pct_change"] = pct("volume")
    results["heads_agree_on_direction"] = bool(
        np.sign(results["cd_head_pct_change"]) == np.sign(results["analytic_pct_change"])
    )
    return results
