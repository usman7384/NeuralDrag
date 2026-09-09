"""Matplotlib-based visualization helpers: pressure fields, parity plots, training curves,
and before/after shape-optimization renders.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


# mesh axes are (x=width, y=height, z=length); remap to (length, width, height) and
# view down the width axis so the car actually looks like a car side-on
_CAR_VIEW = dict(elev=12, azim=-80)


def _to_plot_axes(verts: np.ndarray) -> np.ndarray:
    return verts[:, [2, 0, 1]]


def plot_pressure_field(ax, verts: np.ndarray, faces: np.ndarray, pressure: np.ndarray, title: str, vmin=None, vmax=None):
    pv = _to_plot_axes(verts)
    face_values = pressure[faces].mean(axis=1) if pressure.shape[0] == verts.shape[0] else pressure

    tri = ax.plot_trisurf(
        pv[:, 0], pv[:, 1], pv[:, 2],
        triangles=faces, cmap="coolwarm", linewidth=0, antialiased=True,
    )
    # plot_trisurf ignores a color array passed at construction time -- it must be
    # applied via set_array/set_clim on the returned Poly3DCollection.
    tri.set_array(face_values)
    tri.set_clim(vmin if vmin is not None else face_values.min(), vmax if vmax is not None else face_values.max())

    ax.set_title(title)
    ax.set_box_aspect((np.ptp(pv[:, 0]), np.ptp(pv[:, 1]), np.ptp(pv[:, 2])))
    ax.view_init(**_CAR_VIEW)
    ax.set_axis_off()
    return tri


def render_sample_grid(samples: list[dict], save_path=None, n_cols: int = 4):
    """Grid of raw dataset samples, each colored by its own ground-truth pressure field.

    `samples` is a list of {"id", "pos", "faces", "pressure"} dicts (unnormalized
    verts/pressure straight from src.data.load_mesh / load_pressure).
    """
    n = len(samples)
    n_rows = (n + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(3.2 * n_cols, 2.8 * n_rows))
    for idx, s in enumerate(samples, start=1):
        ax = fig.add_subplot(n_rows, n_cols, idx, projection="3d")
        pv = _to_plot_axes(s["pos"])
        face_values = s["pressure"][s["faces"]].mean(axis=1)
        tri = ax.plot_trisurf(pv[:, 0], pv[:, 1], pv[:, 2], triangles=s["faces"], cmap="coolwarm", linewidth=0, antialiased=True)
        tri.set_array(face_values)
        tri.set_clim(face_values.min(), face_values.max())
        ax.set_title(f'car {s["id"]}', fontsize=9)
        ax.set_box_aspect((np.ptp(pv[:, 0]), np.ptp(pv[:, 1]), np.ptp(pv[:, 2])))
        ax.view_init(**_CAR_VIEW)
        ax.set_axis_off()
    fig.suptitle("Raw dataset samples, colored by ground-truth surface pressure")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_dataset_distributions(pressure_values: np.ndarray, cd_values: np.ndarray, save_path=None):
    """Histograms of per-vertex pressure and per-shape Cd-proxy across the training set."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(pressure_values, bins=80, color="steelblue")
    axes[0].set_title("Per-vertex pressure distribution (train set)")
    axes[0].set_xlabel("pressure")
    axes[0].set_ylabel("count")

    axes[1].hist(cd_values, bins=40, color="indianred")
    axes[1].set_title("Per-shape Cd-proxy distribution (train set)")
    axes[1].set_xlabel("Cd proxy")
    axes[1].set_ylabel("count")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


_MULTI_ANGLES = [
    ("front 3/4", dict(elev=15, azim=-115)),
    ("side", dict(elev=12, azim=-80)),
    ("rear 3/4", dict(elev=15, azim=-45)),
    ("top-down", dict(elev=75, azim=-80)),
]


def render_pressure_multi_angle(verts, faces, pressure, title: str, save_path=None):
    """2x2 grid of the same pressure field from front-3/4, side, rear-3/4, and top-down."""
    fig = plt.figure(figsize=(11, 10))
    for idx, (angle_name, view) in enumerate(_MULTI_ANGLES, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        pv = _to_plot_axes(verts)
        face_values = pressure[faces].mean(axis=1) if pressure.shape[0] == verts.shape[0] else pressure
        tri = ax.plot_trisurf(pv[:, 0], pv[:, 1], pv[:, 2], triangles=faces, cmap="coolwarm", linewidth=0, antialiased=True)
        tri.set_array(face_values)
        tri.set_clim(face_values.min(), face_values.max())
        ax.set_title(angle_name)
        ax.set_box_aspect((np.ptp(pv[:, 0]), np.ptp(pv[:, 1]), np.ptp(pv[:, 2])))
        ax.view_init(**view)
        ax.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def render_pressure_comparison(verts, faces, pressure_true, pressure_pred, save_path=None):
    vmin = min(pressure_true.min(), pressure_pred.min())
    vmax = max(pressure_true.max(), pressure_pred.max())

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    plot_pressure_field(ax1, verts, faces, pressure_true, "Ground truth pressure", vmin, vmax)
    tri = plot_pressure_field(ax2, verts, faces, pressure_pred, "Predicted pressure", vmin, vmax)
    fig.colorbar(tri, ax=[ax1, ax2], shrink=0.6, label="pressure")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_parity(cd_true, cd_pred, save_path=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(cd_true, cd_pred, alpha=0.6, s=20)
    lims = [min(min(cd_true), min(cd_pred)), max(max(cd_true), max(cd_pred))]
    ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
    ax.set_xlabel("True Cd (proxy)")
    ax.set_ylabel("Predicted Cd (proxy)")
    ax.set_title("Parity plot: predicted vs. true Cd")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_training_curves(history: dict, save_path=None):
    """history: dict with keys train_pressure, val_pressure, train_cd, val_cd (lists per epoch)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_pressure"], label="train")
    axes[0].plot(history["val_pressure"], label="val")
    axes[0].set_title("Pressure MSE")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["train_cd"], label="train")
    axes[1].plot(history["val_cd"], label="val")
    axes[1].set_title("Cd MSE")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_cd_over_steps(cd_history: list[float], save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cd_history)
    ax.set_xlabel("optimization step")
    ax.set_ylabel("predicted Cd")
    ax.set_title("Predicted Cd during gradient-based shape optimization")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def exaggerate_displacement(verts_before: np.ndarray, verts_after: np.ndarray, factor: float) -> np.ndarray:
    """Scale up the displacement from `verts_before` for visibility, e.g. because the
    real change is a fraction of a percent of the mesh's size and invisible at normal
    render scale. Standard practice in engineering deformation plots -- always label
    the factor wherever this is used, since the rendered shape is no longer the real
    optimized geometry.
    """
    return verts_before + factor * (verts_after - verts_before)


def render_shape_before_after(verts_before, verts_after, faces, pressure_before=None, pressure_after=None, stats_text: str = None, exaggeration: float = 1.0, save_path=None):
    """Before/after mesh render. If pressure fields are given, color both meshes by
    pressure (shared color scale) instead of a flat color, so the pressure change is
    visible alongside the geometry change. `stats_text` is printed as a caption
    under the figure (e.g. Cd/area numbers) so the improvement isn't left implicit.
    `exaggeration` > 1 scales up the displacement of `verts_after` from `verts_before`
    purely for visibility -- the title says so whenever it's not 1.
    """
    if exaggeration != 1.0:
        verts_after = exaggerate_displacement(verts_before, verts_after, exaggeration)

    fig = plt.figure(figsize=(12, 6.5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    colored = pressure_before is not None and pressure_after is not None
    if colored:
        vmin = float(min(pressure_before.min(), pressure_after.min()))
        vmax = float(max(pressure_before.max(), pressure_after.max()))

    after_title = "Optimized shape" if exaggeration == 1.0 else f"Optimized shape ({exaggeration:.0f}x displacement, for visibility)"

    tri = None
    for ax, verts, pressure, title in [
        (ax1, verts_before, pressure_before, "Original shape"),
        (ax2, verts_after, pressure_after, after_title),
    ]:
        pv = _to_plot_axes(verts)
        if colored:
            face_values = pressure[faces].mean(axis=1)
            tri = ax.plot_trisurf(pv[:, 0], pv[:, 1], pv[:, 2], triangles=faces, cmap="coolwarm", linewidth=0, antialiased=True)
            tri.set_array(face_values)
            tri.set_clim(vmin, vmax)
        else:
            ax.plot_trisurf(pv[:, 0], pv[:, 1], pv[:, 2], triangles=faces, color="lightsteelblue", linewidth=0.1, edgecolor="gray", alpha=0.9)
        ax.set_title(title)
        ax.set_box_aspect((np.ptp(pv[:, 0]), np.ptp(pv[:, 1]), np.ptp(pv[:, 2])))
        ax.view_init(**_CAR_VIEW)
        ax.set_axis_off()

    if colored:
        fig.colorbar(tri, ax=[ax1, ax2], shrink=0.6, label="predicted pressure")
    if stats_text:
        fig.text(0.5, 0.02, stats_text, ha="center", va="bottom", fontsize=10, family="monospace")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def make_morph_gif(pos_history: list[np.ndarray], faces: np.ndarray, save_path: str, pressure_history: list[np.ndarray] = None, cd_over_steps: list[float] = None, stride: int = 5, fps: int = 10, exaggeration: float = 1.0):
    """Morph animation. If `pressure_history` is given, color each frame by that
    step's predicted pressure (shared color scale across all frames) so the pressure
    change is visible, not just the (very subtle) geometry change; `cd_over_steps`
    is shown in the title if given. `exaggeration` > 1 scales up each frame's
    displacement from the first frame, purely for visibility -- see
    `exaggerate_displacement`.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    idxs = list(range(0, len(pos_history), stride))
    pos0 = pos_history[0]
    frames = [_to_plot_axes(exaggerate_displacement(pos0, pos_history[i], exaggeration)) for i in idxs]
    colored = pressure_history is not None
    if colored:
        pressure_frames = [pressure_history[i] for i in idxs]
        vmin = float(min(p.min() for p in pressure_frames))
        vmax = float(max(p.max() for p in pressure_frames))

    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111, projection="3d")

    all_pts = np.concatenate(frames, axis=0)
    lims = [(all_pts[:, i].min(), all_pts[:, i].max()) for i in range(3)]

    def update(frame_idx):
        ax.clear()
        pv = frames[frame_idx]
        step = idxs[frame_idx]
        if colored:
            face_values = pressure_frames[frame_idx][faces].mean(axis=1)
            tri = ax.plot_trisurf(pv[:, 0], pv[:, 1], pv[:, 2], triangles=faces, cmap="coolwarm", linewidth=0, antialiased=True)
            tri.set_array(face_values)
            tri.set_clim(vmin, vmax)
        else:
            ax.plot_trisurf(pv[:, 0], pv[:, 1], pv[:, 2], triangles=faces, color="lightsteelblue", linewidth=0.1, edgecolor="gray", alpha=0.9)
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
        ax.view_init(**_CAR_VIEW)
        ax.set_axis_off()
        title = f"step {step}"
        if cd_over_steps is not None:
            title += f" -- predicted Cd = {cd_over_steps[step]:.2f}"
        ax.set_title(title)

    anim = FuncAnimation(fig, update, frames=len(frames))
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


# interactive plotly versions -- rotatable/zoomable in a browser, unlike the
# fixed-camera matplotlib renders above

def _mesh3d_kwargs(verts: np.ndarray, faces: np.ndarray) -> dict:
    pv = _to_plot_axes(verts)
    return dict(
        x=pv[:, 0], y=pv[:, 1], z=pv[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
    )


def interactive_pressure_comparison(verts, faces, pressure_true, pressure_pred, save_path: str):
    """Side-by-side rotatable 3D pressure fields (vertex-colored), synced camera."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    vmin = float(min(pressure_true.min(), pressure_pred.min()))
    vmax = float(max(pressure_true.max(), pressure_pred.max()))

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "mesh3d"}, {"type": "mesh3d"}]],
        subplot_titles=("Ground truth pressure", "Predicted pressure"),
    )
    for col, (pressure, showscale) in enumerate([(pressure_true, False), (pressure_pred, True)], start=1):
        fig.add_trace(
            go.Mesh3d(
                **_mesh3d_kwargs(verts, faces),
                intensity=pressure, colorscale="RdBu_r", cmin=vmin, cmax=vmax,
                showscale=showscale, colorbar=dict(title="pressure") if showscale else None,
                flatshading=False, lighting=dict(ambient=0.6, diffuse=0.6),
            ),
            row=1, col=col,
        )

    scene_kwargs = dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.update_layout(
        scene=scene_kwargs, scene2=scene_kwargs,
        title="Drag to rotate, scroll to zoom -- ground truth vs. predicted surface pressure",
        margin=dict(l=0, r=0, t=60, b=0), height=550,
    )
    fig.write_html(save_path)
    return fig


def interactive_shape_optimization(pos_history: list[np.ndarray], faces: np.ndarray, cd_over_steps: list[float], save_path: str, pressure_history: list[np.ndarray] = None, stride: int = 2, exaggeration: float = 1.0):
    """Rotatable 3D mesh with a step slider, morphing through the optimization trajectory.

    If `pressure_history` is given, each frame is colored by that step's predicted
    pressure (fixed color scale across all frames) instead of a flat color, since
    the geometry change alone is subtle and the pressure change is the more visible,
    physically meaningful signal. `exaggeration` > 1 scales up each frame's
    displacement from the first frame, purely for visibility.
    """
    import plotly.graph_objects as go

    idxs = list(range(0, len(pos_history), stride))
    if idxs[-1] != len(pos_history) - 1:
        idxs.append(len(pos_history) - 1)

    pos0 = pos_history[0]
    if exaggeration != 1.0:
        pos_history = [exaggerate_displacement(pos0, p, exaggeration) if i in idxs else p for i, p in enumerate(pos_history)]

    all_pts = np.concatenate([_to_plot_axes(pos_history[i]) for i in idxs], axis=0)
    ranges = [(all_pts[:, d].min(), all_pts[:, d].max()) for d in range(3)]

    colored = pressure_history is not None
    if colored:
        pmin = float(min(pressure_history[i].min() for i in idxs))
        pmax = float(max(pressure_history[i].max() for i in idxs))

    def mesh_trace(i, showscale=False):
        kwargs = _mesh3d_kwargs(pos_history[i], faces)
        if colored:
            return go.Mesh3d(**kwargs, intensity=pressure_history[i], colorscale="RdBu_r", cmin=pmin, cmax=pmax,
                              showscale=showscale, colorbar=dict(title="pressure") if showscale else None, flatshading=False)
        return go.Mesh3d(**kwargs, color="lightsteelblue", flatshading=False)

    exag_note = "" if exaggeration == 1.0 else f" (displacement exaggerated {exaggeration:.0f}x for visibility)"

    frames = []
    for i in idxs:
        frames.append(
            go.Frame(
                data=[mesh_trace(i)],
                name=str(i),
                layout=go.Layout(title=f"Step {i} -- predicted Cd = {cd_over_steps[i]:.2f}{exag_note}"),
            )
        )

    fig = go.Figure(
        data=[mesh_trace(0, showscale=colored)],
        frames=frames,
    )
    fig.update_layout(
        title=f"Step 0 -- predicted Cd = {cd_over_steps[0]:.2f}{exag_note}",
        scene=dict(
            aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            xaxis_range=ranges[0], yaxis_range=ranges[1], zaxis_range=ranges[2],
        ),
        margin=dict(l=0, r=0, t=60, b=0), height=650,
        updatemenus=[dict(
            type="buttons", showactive=False,
            buttons=[
                dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)]),
                dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method="animate", args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True))], label=str(i)) for i in idxs],
            currentvalue=dict(prefix="step "),
        )],
    )
    fig.write_html(save_path)
    return fig
