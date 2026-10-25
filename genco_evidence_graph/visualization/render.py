"""Deterministic Matplotlib renderers for GENCO evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from ..provenance import sha256_file
from .model import EvidenceShowcaseSpec, EvidenceShowcaseView, load_showcase_view


RENDER_POLICY_ID = "genco-evidence-showcase-v1"
COLORS = {
    "ink": "#182230",
    "muted": "#5D6B7A",
    "panel": "#F7F9FC",
    "line": "#D8E0EA",
    "purple": "#7756D8",
    "blue": "#2878D0",
    "teal": "#159A9C",
    "orange": "#E88834",
    "green": "#2D9D68",
    "red": "#D84A4A",
    "yellow": "#E8B83F",
    "gray": "#7A8795",
    "amber": "#D98710",
}


@dataclass(frozen=True)
class RenderResult:
    outputs: tuple[Path, ...]
    manifest: Path | None = None


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["line"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["ink"],
            "svg.hashsalt": "genco-evidence-showcase-v1",
        }
    )


def _save(
    figure: Figure,
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    metadata = {
        "Creator": "GencoLens",
        "Description": RENDER_POLICY_ID,
    }
    figure.savefig(
        png,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
        metadata=metadata,
    )
    figure.savefig(
        svg,
        bbox_inches="tight",
        facecolor="white",
        metadata={
            "Creator": "GencoLens",
            "Description": RENDER_POLICY_ID,
            "Date": None,
        },
    )
    plt.close(figure)
    return png, svg


def _footer(figure: Figure, view: EvidenceShowcaseView) -> None:
    figure.text(
        0.01,
        0.012,
        (
            f"Validated RDF: {view.graph_sha256[:12]}…  |  "
            f"Bus artifact: {view.prediction_sha256[:12]}…  |  "
            "Generator artifact: "
            f"{view.generator_prediction_sha256[:12]}…  |  "
            f"View: {RENDER_POLICY_ID}"
        ),
        fontsize=7.5,
        color=COLORS["muted"],
    )


def _badge(
    axes: Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    *,
    width: float = 0.16,
) -> None:
    axes.add_patch(
        FancyBboxPatch(
            (x, y - 0.025),
            width,
            0.05,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=0,
            facecolor=color,
            transform=axes.transAxes,
        )
    )
    axes.text(
        x + width / 2,
        y,
        text,
        color="white",
        weight="bold",
        fontsize=8.5,
        ha="center",
        va="center",
        transform=axes.transAxes,
    )


def render_summary(
    view: EvidenceShowcaseView,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Render the human-readable evidence summary card."""
    _style()
    figure, axes = plt.subplots(figsize=(10.4, 6.2))
    axes.set_axis_off()
    axes.add_patch(
        FancyBboxPatch(
            (0.02, 0.08),
            0.96,
            0.86,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.2,
            edgecolor=COLORS["line"],
            facecolor=COLORS["panel"],
            transform=axes.transAxes,
        )
    )
    axes.text(
        0.06,
        0.88,
        f"Scenario {view.scenario_id}",
        fontsize=24,
        weight="bold",
        transform=axes.transAxes,
    )
    axes.text(
        0.06,
        0.825,
        "GENCO prediction evidence overview",
        fontsize=12,
        color=COLORS["muted"],
        transform=axes.transAxes,
    )
    _badge(
        axes,
        0.79,
        0.87,
        "VALIDATED",
        COLORS["green"],
        width=0.14,
    )
    _badge(
        axes,
        0.77,
        0.79,
        "TRAINING LINK LIMITED",
        COLORS["amber"],
        width=0.20,
    )
    axes.text(
        0.87,
        0.738,
        "production identity unverified",
        fontsize=7.5,
        color=COLORS["amber"],
        ha="center",
        transform=axes.transAxes,
    )

    rows = (
        (
            "Nearest training reference",
            view.references[0].scenario_id,
            COLORS["blue"],
        ),
        (
            "Embedding distance",
            f"{view.nearest_reference_distance:.5f}",
            COLORS["blue"],
        ),
        ("Mean PBE", f"{view.mean_pbe:.6f} MVA", COLORS["teal"]),
        (
            "Maximum PBE",
            f"{view.max_pbe:.6f} MVA  (bus {view.max_pbe_bus})",
            COLORS["red"],
        ),
        (
            "Constraint violations",
            str(view.total_violations),
            COLORS["red"]
            if view.total_violations
            else COLORS["green"],
        ),
    )
    for index, (label, value, color) in enumerate(rows):
        y = 0.72 - index * 0.095
        axes.text(
            0.08,
            y,
            label,
            color=COLORS["muted"],
            fontsize=11,
            transform=axes.transAxes,
        )
        axes.text(
            0.47,
            y,
            value,
            weight="bold",
            fontsize=12,
            color=color,
            transform=axes.transAxes,
        )

    axes.plot(
        [0.06, 0.94],
        [0.225, 0.225],
        color=COLORS["line"],
        linewidth=1,
        transform=axes.transAxes,
    )
    statuses = (
        (
            "Reproduction check",
            view.reproduction_passed,
            "PASSED",
        ),
        (
            "Checkpoint hash",
            view.checkpoint_verified,
            "VERIFIED",
        ),
        (
            "SHACL validation",
            view.shacl_validated,
            "PASSED",
        ),
    )
    for index, (label, passed, text) in enumerate(statuses):
        x = 0.07 + index * 0.30
        axes.text(
            x,
            0.17,
            label,
            fontsize=9,
            color=COLORS["muted"],
            transform=axes.transAxes,
        )
        _badge(
            axes,
            x,
            0.12,
            text if passed else "FAILED",
            COLORS["green"] if passed else COLORS["red"],
            width=0.17,
        )
    _footer(figure, view)
    return _save(figure, output_dir, "evidence_summary")


def _node(
    axes: Axes,
    center: tuple[float, float],
    label: str,
    color: str,
    *,
    width: float = 1.7,
    height: float = 0.72,
    subtitle: str | None = None,
) -> None:
    x, y = center
    axes.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.09",
            linewidth=1.5,
            edgecolor=color,
            facecolor=f"{color}18",
        )
    )
    axes.text(
        x,
        y + (0.08 if subtitle else 0),
        label,
        ha="center",
        va="center",
        fontsize=9.5,
        weight="bold",
        color=color,
    )
    if subtitle:
        axes.text(
            x,
            y - 0.15,
            subtitle,
            ha="center",
            va="center",
            fontsize=7.2,
            color=COLORS["muted"],
        )


def _edge(
    axes: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    start_size: tuple[float, float],
    end_size: tuple[float, float],
) -> None:
    start_point, end_point = _edge_endpoints(
        start,
        end,
        start_size,
        end_size,
    )
    axes.add_patch(
        FancyArrowPatch(
            start_point,
            end_point,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=COLORS["gray"],
            shrinkA=0,
            shrinkB=0,
            connectionstyle="arc3,rad=0.0",
        )
    )
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length = float(np.hypot(delta_x, delta_y))
    normal_x = -delta_y / length
    normal_y = delta_x / length
    axes.text(
        (start_point[0] + end_point[0]) / 2
        + normal_x * 0.13,
        (start_point[1] + end_point[1]) / 2
        + normal_y * 0.13,
        label,
        ha="center",
        va="center",
        fontsize=7,
        color=COLORS["muted"],
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "pad": 1.2,
        },
    )


def _edge_endpoints(
    start: tuple[float, float],
    end: tuple[float, float],
    start_size: tuple[float, float],
    end_size: tuple[float, float],
    *,
    margin: float = 0.045,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Clip a center-to-center edge to two rectangular node boundaries."""
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length = float(np.hypot(delta_x, delta_y))
    if length == 0:
        raise ValueError("Evidence graph edge endpoints must differ")

    def boundary_scale(size: tuple[float, float]) -> float:
        half_width = size[0] / 2
        half_height = size[1] / 2
        scales = []
        if delta_x:
            scales.append(half_width / abs(delta_x))
        if delta_y:
            scales.append(half_height / abs(delta_y))
        return min(scales)

    start_scale = boundary_scale(start_size) + margin / length
    end_scale = boundary_scale(end_size) + margin / length
    if start_scale + end_scale >= 1:
        raise ValueError("Evidence graph nodes overlap along an edge")
    return (
        (
            start[0] + delta_x * start_scale,
            start[1] + delta_y * start_scale,
        ),
        (
            end[0] - delta_x * end_scale,
            end[1] - delta_y * end_scale,
        ),
    )


def render_graph(
    view: EvidenceShowcaseView,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Render a curated, fixed-layout evidence graph."""
    _style()
    figure, axes = plt.subplots(figsize=(14, 8.5))
    axes.set_xlim(0, 12)
    axes.set_ylim(0, 8)
    axes.set_axis_off()
    axes.set_title(
        f"Evidence graph for scenario {view.scenario_id}",
        loc="left",
        pad=18,
        fontsize=20,
        weight="bold",
    )
    axes.text(
        0,
        7.55,
        "A curated view of the validated RDF graph; raw URIs are hidden for clarity.",
        color=COLORS["muted"],
        fontsize=10,
    )
    positions = {
        "dataset": (1.1, 5.9),
        "checkpoint": (1.1, 3.9),
        "training": (1.1, 2.35),
        "scenario": (3.7, 0.8),
        "inference": (4.0, 3.9),
        "prediction": (6.8, 3.9),
        "bus_prediction_artifact": (5.6, 6.15),
        "generator_prediction_artifact": (8.0, 6.15),
        "physical": (10.5, 6.5),
        "violations": (10.5, 5.0),
        "embedding": (10.5, 3.5),
        "references": (10.5, 2.0),
        "reproduction": (6.8, 1.0),
    }
    sizes = {
        "dataset": (1.7, 0.72),
        "checkpoint": (1.7, 0.72),
        "training": (2.1, 0.72),
        "scenario": (1.7, 0.72),
        "inference": (1.7, 0.72),
        "prediction": (1.9, 0.72),
        "bus_prediction_artifact": (2.1, 0.72),
        "generator_prediction_artifact": (2.3, 0.72),
        "physical": (1.7, 0.72),
        "violations": (1.7, 0.72),
        "embedding": (1.7, 0.72),
        "references": (1.7, 0.72),
        "reproduction": (1.9, 0.72),
    }
    _node(
        axes,
        positions["dataset"],
        "Test dataset",
        COLORS["teal"],
        subtitle=view.dataset_revision[:10] + "…",
    )
    _node(
        axes,
        positions["checkpoint"],
        "Model checkpoint",
        COLORS["blue"],
        subtitle=view.checkpoint_sha256[:10] + "…",
    )
    _node(
        axes,
        positions["training"],
        "Training association",
        COLORS["amber"],
        subtitle="production identity unverified",
        width=2.1,
    )
    _node(
        axes,
        positions["scenario"],
        f"Scenario {view.scenario_id}",
        COLORS["teal"],
    )
    _node(
        axes,
        positions["inference"],
        "Inference run",
        COLORS["orange"],
        subtitle=view.inference_run_id[:10] + "…",
    )
    _node(
        axes,
        positions["prediction"],
        "Prediction",
        COLORS["purple"],
        width=1.9,
    )
    _node(
        axes,
        positions["bus_prediction_artifact"],
        "Bus prediction artifact",
        COLORS["gray"],
        subtitle=view.prediction_sha256[:10] + "…",
        width=2.1,
    )
    _node(
        axes,
        positions["generator_prediction_artifact"],
        "Generator prediction artifact",
        COLORS["gray"],
        subtitle=view.generator_prediction_sha256[:10] + "…",
        width=2.3,
    )
    _node(
        axes,
        positions["physical"],
        "Physical evidence",
        COLORS["green"],
        subtitle=f"max PBE {view.max_pbe:.4g}",
    )
    _node(
        axes,
        positions["violations"],
        "Constraint violations",
        COLORS["red"],
        subtitle=f"count {view.total_violations}",
    )
    _node(
        axes,
        positions["embedding"],
        "Embedding evidence",
        COLORS["blue"],
        subtitle=f"distance {view.nearest_reference_distance:.4f}",
    )
    _node(
        axes,
        positions["references"],
        "5 nearest references",
        COLORS["yellow"],
        subtitle=f"closest {view.references[0].scenario_id}",
    )
    _node(
        axes,
        positions["reproduction"],
        "Reproduction check",
        COLORS["green"],
        subtitle="passed",
        width=1.9,
    )
    for start, end, label in (
        ("dataset", "inference", "used"),
        ("checkpoint", "inference", "used"),
        (
            "checkpoint",
            "training",
            "associated evidence",
        ),
        ("scenario", "inference", "used"),
        ("inference", "prediction", "generated"),
        (
            "prediction",
            "bus_prediction_artifact",
            "has provenance artifact",
        ),
        (
            "prediction",
            "generator_prediction_artifact",
            "has provenance artifact",
        ),
        ("prediction", "physical", "has physical evidence"),
        ("physical", "violations", "reports"),
        ("prediction", "embedding", "has embedding evidence"),
        ("embedding", "references", "nearest cases"),
        ("reproduction", "prediction", "reproduces"),
    ):
        _edge(
            axes,
            positions[start],
            positions[end],
            label,
            sizes[start],
            sizes[end],
        )
    _footer(figure, view)
    return _save(figure, output_dir, "evidence_graph")


def _chart_title(
    axes: Axes,
    title: str,
    subtitle: str,
) -> None:
    axes.set_title(title, loc="left", weight="bold", pad=18)
    axes.text(
        0,
        1.02,
        subtitle,
        transform=axes.transAxes,
        color=COLORS["muted"],
        fontsize=8.5,
    )
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", color=COLORS["line"], linewidth=0.7)
    axes.set_axisbelow(True)


def _reference_chart(
    view: EvidenceShowcaseView,
    output_dir: Path,
) -> tuple[Path, ...]:
    figure, axes = plt.subplots(figsize=(9, 5.4))
    labels = [value.scenario_id for value in view.references]
    distances = [value.distance for value in view.references]
    colors = [
        COLORS["purple"],
        *[COLORS["blue"]] * (len(labels) - 1),
    ]
    bars = axes.barh(labels[::-1], distances[::-1], color=colors[::-1])
    axes.set_xlabel("Euclidean distance in pooled embedding space")
    _chart_title(
        axes,
        "Nearest training references",
        "Lower distance indicates greater similarity to the target representation.",
    )
    lower = min(distances) * 0.92
    axes.set_xlim(lower, max(distances) * 1.08)
    for bar, value in zip(bars, distances[::-1], strict=True):
        axes.text(
            value + (max(distances) - lower) * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.5f}",
            va="center",
            fontsize=9,
        )
    _footer(figure, view)
    return _save(figure, output_dir, "nearest_references")


def _pbe_chart(
    view: EvidenceShowcaseView,
    output_dir: Path,
) -> tuple[Path, ...]:
    figure, axes = plt.subplots(figsize=(10, 5.4))
    buses = view.bus_predictions["bus"].to_numpy()
    pbe = view.bus_predictions["PBE"].to_numpy()
    colors = [
        COLORS["red"]
        if bus == view.max_pbe_bus
        else COLORS["teal"]
        for bus in buses
    ]
    axes.bar(buses, pbe, color=colors)
    axes.axhline(
        view.mean_pbe,
        color=COLORS["orange"],
        linestyle="--",
        linewidth=1.5,
        label=f"Mean PBE {view.mean_pbe:.6f}",
    )
    axes.set_xticks(buses)
    axes.set_xlabel("Bus index")
    axes.set_ylabel("PBE (MVA)")
    axes.legend(frameon=False)
    _chart_title(
        axes,
        "Power-balance error by bus",
        f"Bus {view.max_pbe_bus} has the maximum PBE ({view.max_pbe:.6f} MVA).",
    )
    _footer(figure, view)
    return _save(figure, output_dir, "bus_pbe")


def _prediction_chart(
    view: EvidenceShowcaseView,
    output_dir: Path,
) -> tuple[Path, ...]:
    figure, (bus_axes, gen_axes) = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
    )
    buses = view.bus_predictions["bus"].to_numpy()
    bus_axes.plot(
        buses,
        view.bus_predictions["Vm_target"],
        marker="o",
        color=COLORS["blue"],
        label="Target",
    )
    bus_axes.plot(
        buses,
        view.bus_predictions["Vm_pred"],
        marker="s",
        color=COLORS["orange"],
        label="Prediction",
    )
    bus_axes.set_xlabel("Bus index")
    bus_axes.set_ylabel("Voltage magnitude (p.u.)")
    bus_axes.set_xticks(buses)
    bus_axes.legend(frameon=False)
    _chart_title(
        bus_axes,
        "Bus voltage magnitude",
        "Prediction and target values across all 14 buses.",
    )

    generators = view.generator_predictions["idx"].to_numpy()
    width = 0.34
    gen_axes.bar(
        generators - width / 2,
        view.generator_predictions["p_mw_target"],
        width,
        color=COLORS["blue"],
        label="Target",
    )
    gen_axes.bar(
        generators + width / 2,
        view.generator_predictions["p_mw_pred"],
        width,
        color=COLORS["orange"],
        label="Prediction",
    )
    gen_axes.set_xlabel("Generator index")
    gen_axes.set_ylabel("Active power (MW)")
    gen_axes.set_xticks(generators)
    gen_axes.legend(frameon=False)
    _chart_title(
        gen_axes,
        "Generator active power",
        "Prediction and target values for scenario generators.",
    )
    figure.suptitle(
        f"Prediction agreement for scenario {view.scenario_id}",
        x=0.06,
        y=1.03,
        ha="left",
        fontsize=17,
        weight="bold",
    )
    figure.tight_layout()
    _footer(figure, view)
    return _save(figure, output_dir, "prediction_comparison")


def _violation_chart(
    view: EvidenceShowcaseView,
    output_dir: Path,
) -> tuple[Path, ...]:
    figure, axes = plt.subplots(figsize=(9.5, 4.8))
    labels = (
        "Voltage limits",
        "Reactive power",
        "Active power",
    )
    counts = (
        view.voltage_violations,
        view.reactive_violations,
        view.active_power_violations,
    )
    colors = [
        COLORS["red"] if value else COLORS["green"]
        for value in counts
    ]
    bars = axes.barh(labels[::-1], counts[::-1], color=colors[::-1])
    axes.set_xlabel("Violation count")
    axes.set_xticks(
        range(0, max(2, max(counts) + 1))
    )
    _chart_title(
        axes,
        "Constraint checks",
        (
            f"{view.total_violations} total violation"
            f"{'' if view.total_violations == 1 else 's'}; "
            "green categories have no detected violation."
        ),
    )
    for bar, value in zip(bars, counts[::-1], strict=True):
        axes.text(
            value + 0.03,
            bar.get_y() + bar.get_height() / 2,
            "PASS" if value == 0 else str(value),
            va="center",
            weight="bold",
            color=COLORS["green"]
            if value == 0
            else COLORS["red"],
        )
    _footer(figure, view)
    return _save(figure, output_dir, "constraint_violations")


def _reproduction_chart(
    view: EvidenceShowcaseView,
    output_dir: Path,
) -> tuple[Path, ...]:
    figure, axes = plt.subplots(figsize=(10, 5.2))
    labels = ("Voltage fields", "Power/residual fields")
    observed = np.array(
        [
            view.maximum_voltage_difference,
            view.maximum_power_difference,
        ]
    )
    allowed = np.array(
        [view.voltage_tolerance, view.power_tolerance]
    )
    ratios = observed / allowed
    y = np.arange(2)
    axes.barh(
        y,
        np.ones(2),
        color="#E9EEF5",
        height=0.45,
    )
    axes.barh(
        y,
        ratios,
        color=[COLORS["green"], COLORS["green"]],
        height=0.45,
    )
    axes.axvline(
        1,
        color=COLORS["red"],
        linestyle="--",
        linewidth=1.4,
        label="Allowed limit",
    )
    axes.set_yticks(y, labels)
    axes.set_xlim(0, 1.12)
    axes.set_xlabel("Fraction of allowed absolute tolerance")
    axes.xaxis.set_major_formatter(
        lambda value, _position: f"{value * 100:.0f}%"
    )
    axes.legend(frameon=False, loc="lower right")
    _chart_title(
        axes,
        "Reproduction equivalence",
        (
            f"Policy {view.tolerance_policy_id}; both observed maxima "
            "remain within their approved limits."
        ),
    )
    for index, (actual, limit, ratio) in enumerate(
        zip(observed, allowed, ratios, strict=True)
    ):
        axes.text(
            min(ratio + 0.02, 0.86),
            index,
            f"{actual:.3e} / {limit:.3e}  PASS",
            va="center",
            fontsize=9,
            weight="bold",
            color=COLORS["ink"],
        )
    _footer(figure, view)
    return _save(figure, output_dir, "reproduction_check")


def render_charts(
    view: EvidenceShowcaseView,
    output_dir: Path,
) -> tuple[Path, ...]:
    """Render all numeric evidence charts."""
    _style()
    outputs: list[Path] = []
    for renderer in (
        _reference_chart,
        _pbe_chart,
        _prediction_chart,
        _violation_chart,
        _reproduction_chart,
    ):
        outputs.extend(renderer(view, output_dir))
    return tuple(outputs)


def _manifest(
    spec: EvidenceShowcaseSpec,
    view: EvidenceShowcaseView,
    outputs: tuple[Path, ...],
) -> dict[str, object]:
    project_root = spec.project_root.resolve()

    def portable_path(path: Path) -> str:
        try:
            return path.resolve().relative_to(
                project_root
            ).as_posix()
        except ValueError:
            return str(path.resolve())

    return {
        "format_version": 1,
        "artifact_type": "evidence_showcase_figure_bundle",
        "render_policy_id": RENDER_POLICY_ID,
        "scenario_id": view.scenario_id,
        "presentation_only": True,
        "source_artifacts": [
            {"path": path, "sha256": digest}
            for path, digest in view.artifact_hashes
        ],
        "shapes": {
            "path": portable_path(spec.shapes),
            "sha256": sha256_file(spec.shapes),
        },
        "verification": {
            "shacl_validated": view.shacl_validated,
            "checkpoint_verified": view.checkpoint_verified,
            "prediction_artifacts_verified": True,
        },
        "figures": [
            {
                "path": portable_path(path),
                "sha256": sha256_file(path),
            }
            for path in sorted(outputs)
        ],
    }


def render_showcase(
    spec: EvidenceShowcaseSpec,
    sections: tuple[str, ...] = (
        "summary",
        "graph",
        "charts",
    ),
) -> RenderResult:
    """Verify evidence and render selected showcase sections."""
    allowed = {"summary", "graph", "charts"}
    unknown = set(sections) - allowed
    if unknown:
        raise ValueError(
            f"Unknown showcase sections: {sorted(unknown)}"
        )
    output_dir = spec.output_dir.resolve()
    view = load_showcase_view(spec)
    renderers: dict[
        str,
        Callable[
            [EvidenceShowcaseView, Path],
            tuple[Path, ...],
        ],
    ] = {
        "summary": render_summary,
        "graph": render_graph,
        "charts": render_charts,
    }
    outputs: list[Path] = []
    for section in sections:
        outputs.extend(
            renderers[section](view, output_dir)
        )
    manifest_name = (
        "figure_manifest.json"
        if set(sections) == {"summary", "graph", "charts"}
        else f"figure_manifest_{'_'.join(sections)}.json"
    )
    manifest_path = output_dir / manifest_name
    manifest_path.write_text(
        json.dumps(
            _manifest(spec, view, tuple(outputs)),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return RenderResult(
        outputs=tuple(outputs),
        manifest=manifest_path,
    )
