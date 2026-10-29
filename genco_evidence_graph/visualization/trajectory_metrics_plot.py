"""Reusable static visualizations for GENCO latent trajectory metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd


GROUP_ORDER = (
    "familiar_good",
    "unfamiliar_good",
    "familiar_bad",
    "unfamiliar_bad",
)
GROUP_LABELS = {
    "familiar_good": "Familiar, good",
    "unfamiliar_good": "Unfamiliar, good",
    "familiar_bad": "Familiar, bad",
    "unfamiliar_bad": "Unfamiliar, bad",
}
GROUP_TICK_LABELS = {
    key: value.replace(", ", "\n")
    for key, value in GROUP_LABELS.items()
}
GROUP_COLORS = {
    "familiar_good": "#2B7A4B",
    "unfamiliar_good": "#277DA1",
    "familiar_bad": "#E07A1F",
    "unfamiliar_bad": "#C44536",
}
SELECTION_MARKERS = {
    "corner": "o",
    "inner_boundary": "s",
    "physical_diversity": "D",
}
SELECTION_LABELS = {
    "corner": "Corner",
    "inner_boundary": "Inner boundary",
    "physical_diversity": "Physical diversity",
}
ENTITY_MARKERS = {"bus": "o", "generator": "^"}
ENTITY_COLORS = {"bus": "#276FBF", "generator": "#8E5AB5"}
LOW_PERCENTILE = 0.25
HIGH_PERCENTILE = 0.75
OUTPUT_FORMATS = ("png", "pdf")


@dataclass(frozen=True)
class TrajectoryPlotSpec:
    """Input artifacts and output directory for the figure package."""

    scores: Path
    cohort: Path
    entity_metrics: Path
    step_metrics: Path
    scenario_summary: Path
    bus_predictions: Path
    output_dir: Path = Path("figure/trajectory_metrics")
    dpi: int = 220

    def __post_init__(self) -> None:
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")


@dataclass(frozen=True)
class TrajectoryPlotData:
    """Validated tables and shared derived values used by all figures."""

    scores: pd.DataFrame
    cohort: pd.DataFrame
    entities: pd.DataFrame
    steps: pd.DataFrame
    scenarios: pd.DataFrame
    bus_predictions: pd.DataFrame
    scenario_order: tuple[int, ...]


def _require_columns(
    table: pd.DataFrame,
    required: Iterable[str],
    label: str,
) -> None:
    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise ValueError(f"{label} lacks required columns: {missing}")


def _read_parquet(
    path: Path,
    columns: list[str],
    label: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    table = pd.read_parquet(path, columns=columns)
    _require_columns(table, columns, label)
    return table


def load_trajectory_plot_data(
    spec: TrajectoryPlotSpec,
) -> TrajectoryPlotData:
    """Load and validate all tables once for reuse across the figures."""
    scores = _read_parquet(
        spec.scores,
        [
            "scenario_id",
            "nearest_reference_distance",
            "mean_absolute_pbe",
        ],
        "Scenario score table",
    )
    cohort = _read_parquet(
        spec.cohort,
        [
            "scenario_id",
            "distance_percentile",
            "residual_percentile",
            "cohort_group",
            "selection_reason",
            "selection_rank",
            "max_pbe_bus",
        ],
        "Selected cohort table",
    )
    entities = _read_parquet(
        spec.entity_metrics,
        [
            "scenario_id",
            "entity_type",
            "entity_id",
            "path_length",
            "net_displacement",
            "path_efficiency",
            "local_directional_consistency",
            "global_directional_alignment",
            "cohort_group",
            "path_length_rank_in_entity_type",
            "path_length_share_in_entity_type",
        ],
        "Entity trajectory metric table",
    )
    steps = _read_parquet(
        spec.step_metrics,
        [
            "scenario_id",
            "entity_type",
            "entity_id",
            "start_step",
            "end_step",
            "step_movement",
            "cohort_group",
        ],
        "Step trajectory metric table",
    )
    scenarios = _read_parquet(
        spec.scenario_summary,
        [
            "scenario_id",
            "cohort_group",
            "entity_type",
            "total_path_length",
            "mean_path_length",
            "mean_net_displacement",
            "mean_path_efficiency",
            "mean_local_directional_consistency",
            "mean_global_directional_alignment",
        ],
        "Scenario trajectory summary table",
    )
    bus_predictions = _read_parquet(
        spec.bus_predictions,
        ["scenario", "bus", "PBE"],
        "Bus prediction table",
    ).rename(columns={"scenario": "scenario_id", "bus": "entity_id"})

    for table in (
        scores,
        cohort,
        entities,
        steps,
        scenarios,
        bus_predictions,
    ):
        table["scenario_id"] = pd.to_numeric(
            table["scenario_id"],
            errors="raise",
        ).astype("int64")

    if cohort["scenario_id"].duplicated().any():
        raise ValueError("Selected cohort contains duplicate scenarios")
    unknown_groups = sorted(set(cohort["cohort_group"]) - set(GROUP_ORDER))
    if unknown_groups:
        raise ValueError(f"Unsupported cohort groups: {unknown_groups}")
    unknown_reasons = sorted(
        set(cohort["selection_reason"]) - set(SELECTION_MARKERS)
    )
    if unknown_reasons:
        raise ValueError(
            f"Unsupported selection reasons: {unknown_reasons}"
        )

    cohort_ids = set(cohort["scenario_id"])
    for table, label in (
        (entities, "Entity metrics"),
        (steps, "Step metrics"),
        (scenarios, "Scenario summaries"),
    ):
        if set(table["scenario_id"]) != cohort_ids:
            raise ValueError(
                f"{label} scenario IDs disagree with the cohort"
            )
    prediction_ids = set(bus_predictions["scenario_id"])
    if not cohort_ids <= prediction_ids:
        missing = sorted(cohort_ids - prediction_ids)
        raise ValueError(
            f"Bus predictions lack cohort scenarios: {missing}"
        )
    bus_predictions = bus_predictions[
        bus_predictions["scenario_id"].isin(cohort_ids)
    ].copy()

    cohort_group_by_scenario = cohort.set_index("scenario_id")[
        "cohort_group"
    ]
    for table, label in (
        (entities, "Entity metrics"),
        (steps, "Step metrics"),
        (scenarios, "Scenario summaries"),
    ):
        observed = table.set_index("scenario_id")["cohort_group"]
        expected = observed.index.map(cohort_group_by_scenario)
        if not np.array_equal(observed.to_numpy(), expected.to_numpy()):
            raise ValueError(f"{label} cohort groups disagree")

    scores["distance_percentile"] = scores[
        "nearest_reference_distance"
    ].rank(method="average", pct=True)
    scores["residual_percentile"] = scores[
        "mean_absolute_pbe"
    ].rank(method="average", pct=True)

    group_rank = {group: index for index, group in enumerate(GROUP_ORDER)}
    ordered = cohort.assign(
        _group_rank=cohort["cohort_group"].map(group_rank)
    ).sort_values(
        ["_group_rank", "selection_rank", "scenario_id"],
        kind="stable",
    )
    scenario_order = tuple(ordered["scenario_id"].astype(int))

    return TrajectoryPlotData(
        scores=scores,
        cohort=cohort,
        entities=entities,
        steps=steps,
        scenarios=scenarios,
        bus_predictions=bus_predictions,
        scenario_order=scenario_order,
    )


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def _panel_label(axis: Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def _group_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=GROUP_COLORS[group],
            marker="o",
            linestyle="-",
            label=GROUP_LABELS[group],
        )
        for group in GROUP_ORDER
    ]


def _export_figure(
    figure: Figure,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in OUTPUT_FORMATS:
        path = output_dir / f"{stem}.{suffix}"
        save_options: dict[str, object] = {
            "bbox_inches": "tight",
            "facecolor": "white",
        }
        if suffix == "png":
            save_options["dpi"] = dpi
        figure.savefig(path, **save_options)
        paths.append(path)
    plt.close(figure)
    return tuple(paths)


def plot_metric_intuition(data: TrajectoryPlotData) -> Figure:
    """Illustrate trajectory snapshots, movement, and path efficiency."""
    del data
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    figure.suptitle(
        "Internal trajectory and metric intuition",
        fontsize=14,
        fontweight="bold",
    )

    points = np.array(
        [
            [0.08, 0.18],
            [0.21, 0.50],
            [0.39, 0.36],
            [0.50, 0.73],
            [0.68, 0.58],
            [0.88, 0.84],
        ]
    )
    axis = axes[0]
    axis.plot(
        points[:, 0],
        points[:, 1],
        color="#5A5A5A",
        linewidth=1.7,
        zorder=1,
    )
    axis.scatter(
        points[:, 0],
        points[:, 1],
        c=np.arange(len(points)),
        cmap="viridis",
        s=70,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                color="#5A5A5A",
                linewidth=1.2,
            )
        )
        midpoint = (start + end) / 2
        axis.text(
            midpoint[0],
            midpoint[1] + 0.035,
            rf"$d_{{{index + 1}}}$",
            ha="center",
            fontsize=9,
        )
    axis.plot(
        points[[0, -1], 0],
        points[[0, -1], 1],
        linestyle="--",
        color="#C44536",
        linewidth=1.5,
        label="Net displacement",
    )
    axis.text(
        points[0, 0],
        points[0, 1] - 0.08,
        "Layer 1",
        ha="center",
    )
    axis.text(
        points[-1, 0],
        points[-1, 1] + 0.05,
        "Layer $L$",
        ha="center",
    )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Illustrative latent coordinate 1",
        ylabel="Illustrative latent coordinate 2",
        title="Layer-wise latent trajectory",
    )
    axis.legend(loc="lower right")
    _panel_label(axis, "a")

    axis = axes[1]
    axis.axis("off")
    definitions = (
        (
            "Step movement",
            r"$d_t=\Vert z_{t+1}-z_t\Vert_2$",
            "Movement between adjacent layer snapshots",
        ),
        (
            "Path length",
            r"$P=\sum_t d_t$",
            "Total distance traveled through latent space",
        ),
        (
            "Net displacement",
            r"$D=\Vert z_L-z_1\Vert_2$",
            "Straight-line distance from first to final state",
        ),
        (
            "Path efficiency",
            r"$E=D/P$",
            "Directness of the trajectory, bounded by 0 and 1",
        ),
    )
    y_positions = np.linspace(0.85, 0.16, len(definitions))
    for y, (name, equation, explanation) in zip(
        y_positions,
        definitions,
    ):
        axis.add_patch(
            Rectangle(
                (0.02, y - 0.10),
                0.96,
                0.18,
                transform=axis.transAxes,
                facecolor="#F6F7F8",
                edgecolor="#D0D3D5",
                linewidth=0.8,
            )
        )
        axis.text(
            0.05,
            y + 0.035,
            name,
            transform=axis.transAxes,
            fontweight="bold",
            va="center",
        )
        axis.text(
            0.95,
            y + 0.035,
            equation,
            transform=axis.transAxes,
            ha="right",
            va="center",
            fontsize=11,
        )
        axis.text(
            0.05,
            y - 0.045,
            explanation,
            transform=axis.transAxes,
            color="#555555",
            va="center",
        )
    axis.set_title("Core geometric quantities", pad=12)
    _panel_label(axis, "b")
    figure.tight_layout()
    return figure


def plot_cohort_selection(data: TrajectoryPlotData) -> Figure:
    """Plot the percentile map and selected cohort roles."""
    figure, axis = plt.subplots(figsize=(8.2, 7.0))
    corner_regions = (
        (0.0, 0.0, "familiar_good"),
        (0.75, 0.0, "unfamiliar_good"),
        (0.0, 0.75, "familiar_bad"),
        (0.75, 0.75, "unfamiliar_bad"),
    )
    for x, y, group in corner_regions:
        axis.add_patch(
            Rectangle(
                (x, y),
                0.25,
                0.25,
                facecolor=GROUP_COLORS[group],
                edgecolor="none",
                alpha=0.10,
                zorder=0,
            )
        )
    axis.scatter(
        data.scores["distance_percentile"],
        data.scores["residual_percentile"],
        s=7,
        color="#888888",
        alpha=0.16,
        linewidths=0,
        rasterized=True,
        label="Test scenarios",
    )
    for reason, marker in SELECTION_MARKERS.items():
        selected = data.cohort[data.cohort["selection_reason"] == reason]
        axis.scatter(
            selected["distance_percentile"],
            selected["residual_percentile"],
            c=selected["cohort_group"].map(GROUP_COLORS),
            marker=marker,
            s=70,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
            label=SELECTION_LABELS[reason],
        )
    non_corner = data.cohort[
        data.cohort["selection_reason"] != "corner"
    ]
    for row in non_corner.itertuples(index=False):
        x_offset = -5 if row.distance_percentile > 0.5 else 5
        alignment = "right" if x_offset < 0 else "left"
        y_offset = -7 if row.residual_percentile > 0.5 else 6
        axis.annotate(
            str(row.scenario_id),
            (row.distance_percentile, row.residual_percentile),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=6.5,
            ha=alignment,
            color="#333333",
        )
    corner_label_positions = {
        "familiar_good": (0.055, 0.075),
        "unfamiliar_good": (0.84, 0.115),
        "familiar_bad": (0.07, 0.865),
        "unfamiliar_bad": (0.84, 0.91),
    }
    for group, position in corner_label_positions.items():
        corner = data.cohort[
            (data.cohort["cohort_group"] == group)
            & (data.cohort["selection_reason"] == "corner")
        ].sort_values("selection_rank")
        label = "Corner: " + ", ".join(
            corner["scenario_id"].astype(str)
        )
        axis.text(
            *position,
            label,
            transform=axis.transAxes,
            fontsize=6.5,
            color=GROUP_COLORS[group],
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": GROUP_COLORS[group],
                "alpha": 0.85,
                "linewidth": 0.7,
            },
        )
    for threshold in (LOW_PERCENTILE, HIGH_PERCENTILE):
        axis.axvline(
            threshold,
            color="#555555",
            linestyle="--",
            linewidth=0.9,
        )
        axis.axhline(
            threshold,
            color="#555555",
            linestyle="--",
            linewidth=0.9,
        )
    axis.set(
        xlim=(-0.015, 1.015),
        ylim=(-0.015, 1.015),
        xlabel="Nearest-reference distance percentile",
        ylabel="Mean absolute PBE percentile",
        title="Percentile cohort-selection map",
    )
    group_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=GROUP_COLORS[group],
            markeredgecolor="white",
            markersize=7,
            label=GROUP_LABELS[group],
        )
        for group in GROUP_ORDER
    ]
    role_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor="#666666",
            markeredgecolor="white",
            markersize=7,
            label=SELECTION_LABELS[reason],
        )
        for reason, marker in SELECTION_MARKERS.items()
    ]
    axis.legend(
        handles=group_handles + role_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    figure.tight_layout()
    return figure


def plot_group_metric_comparison(data: TrajectoryPlotData) -> Figure:
    """Compare scenario-balanced trajectory metrics across four groups."""
    metrics = (
        ("total_path_length", "Total path length", ".1f"),
        ("mean_path_length", "Mean entity path length", ".1f"),
        ("mean_net_displacement", "Mean net displacement", ".1f"),
        ("mean_path_efficiency", "Path efficiency", ".2f"),
        (
            "mean_local_directional_consistency",
            "Local directional consistency",
            ".2f",
        ),
        (
            "mean_global_directional_alignment",
            "Global directional alignment",
            ".2f",
        ),
    )
    selected = data.scenarios[data.scenarios["entity_type"] == "all"]
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 7.3))
    figure.suptitle(
        "Four-group trajectory metric comparison",
        fontsize=14,
        fontweight="bold",
    )
    jitter = np.linspace(-0.10, 0.10, 5)
    for axis, (metric, title, value_format), panel in zip(
        axes.flat,
        metrics,
        "abcdef",
    ):
        for group_index, group in enumerate(GROUP_ORDER):
            values = (
                selected[selected["cohort_group"] == group]
                .sort_values("scenario_id")[metric]
                .to_numpy(dtype=float)
            )
            offsets = jitter[: len(values)]
            axis.scatter(
                group_index + offsets,
                values,
                color=GROUP_COLORS[group],
                s=30,
                alpha=0.72,
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
            )
            mean = float(values.mean())
            axis.scatter(
                group_index,
                mean,
                marker="D",
                color=GROUP_COLORS[group],
                edgecolor="#222222",
                linewidth=0.7,
                s=58,
                zorder=3,
            )
            axis.annotate(
                format(mean, value_format),
                (group_index, mean),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                fontweight="bold",
            )
        axis.set_xticks(
            range(len(GROUP_ORDER)),
            [GROUP_TICK_LABELS[group] for group in GROUP_ORDER],
        )
        axis.set_title(title)
        axis.tick_params(axis="x", labelsize=8)
        _panel_label(axis, panel)
    figure.text(
        0.5,
        0.015,
        "Small points: individual scenarios     Diamonds: "
        "unweighted group means",
        ha="center",
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    return figure


def _scenario_transition_shares(steps: pd.DataFrame) -> pd.DataFrame:
    scenario_steps = (
        steps.groupby(
            ["scenario_id", "cohort_group", "start_step", "end_step"],
            sort=True,
            as_index=False,
        )["step_movement"]
        .sum()
        .rename(columns={"step_movement": "transition_movement"})
    )
    totals = scenario_steps.groupby("scenario_id")[
        "transition_movement"
    ].transform("sum")
    if (totals <= 0).any():
        raise ValueError("Transition movement totals must be positive")
    scenario_steps["movement_share"] = (
        scenario_steps["transition_movement"] / totals
    )
    return scenario_steps


def plot_transition_profiles(data: TrajectoryPlotData) -> Figure:
    """Plot scenario-balanced movement shares across layer transitions."""
    shares = _scenario_transition_shares(data.steps)
    profile = (
        shares.groupby(
            ["cohort_group", "start_step", "end_step"],
            sort=True,
        )["movement_share"]
        .agg(["mean", "std"])
        .reset_index()
        .fillna({"std": 0.0})
    )
    figure, axis = plt.subplots(figsize=(10.5, 5.4))
    for group in GROUP_ORDER:
        selected = profile[profile["cohort_group"] == group]
        x = selected["start_step"].to_numpy(dtype=float)
        mean = 100 * selected["mean"].to_numpy(dtype=float)
        std = 100 * selected["std"].to_numpy(dtype=float)
        color = GROUP_COLORS[group]
        axis.plot(
            x,
            mean,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            label=GROUP_LABELS[group],
        )
        axis.fill_between(
            x,
            np.maximum(0, mean - std),
            mean + std,
            color=color,
            alpha=0.10,
            linewidth=0,
        )
    axis.set_xticks(
        range(1, 12),
        [f"{step}\u2192{step + 1}" for step in range(1, 12)],
    )
    axis.set(
        xlabel="Layer transition",
        ylabel="Mean share of scenario movement (%)",
        title="Layer-transition movement profiles",
    )
    axis.legend(ncol=2)
    axis.text(
        0.99,
        0.98,
        "Bands show \u00b11 SD across scenarios",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout()
    return figure


def plot_path_efficiency(data: TrajectoryPlotData) -> Figure:
    """Relate entity path length to path efficiency."""
    figure, axis = plt.subplots(figsize=(9.2, 6.4))
    for group in GROUP_ORDER:
        for entity_type in ("bus", "generator"):
            selected = data.entities[
                (data.entities["cohort_group"] == group)
                & (data.entities["entity_type"] == entity_type)
            ]
            axis.scatter(
                selected["path_length"],
                selected["path_efficiency"],
                color=GROUP_COLORS[group],
                marker=ENTITY_MARKERS[entity_type],
                s=36 if entity_type == "bus" else 45,
                alpha=0.58,
                edgecolor="white",
                linewidth=0.5,
                rasterized=True,
            )
    notable = (
        data.entities.sort_values(
            ["cohort_group", "path_length"],
            ascending=[True, False],
            kind="stable",
        )
        .groupby("cohort_group", sort=False)
        .head(1)
    )
    for row in notable.itertuples(index=False):
        axis.annotate(
            f"{row.scenario_id}: {row.entity_type[0].upper()}{row.entity_id}",
            (row.path_length, row.path_efficiency),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            color=GROUP_COLORS[row.cohort_group],
        )
    efficiencies = data.entities["path_efficiency"].to_numpy(dtype=float)
    efficiency_padding = max(
        0.03,
        0.08 * float(np.ptp(efficiencies)),
    )
    axis.set(
        xlabel="Entity path length",
        ylabel="Path efficiency",
        ylim=(
            max(0.0, float(efficiencies.min()) - efficiency_padding),
            min(1.0, float(efficiencies.max()) + efficiency_padding),
        ),
        title="Path length versus path efficiency",
    )
    group_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=GROUP_COLORS[group],
            markeredgecolor="none",
            markersize=7,
            label=GROUP_LABELS[group],
        )
        for group in GROUP_ORDER
    ]
    entity_handles = [
        Line2D(
            [0],
            [0],
            marker=ENTITY_MARKERS[entity_type],
            linestyle="none",
            markerfacecolor="#777777",
            markeredgecolor="white",
            markersize=7,
            label=entity_type.title(),
        )
        for entity_type in ("bus", "generator")
    ]
    axis.legend(
        handles=group_handles + entity_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    figure.tight_layout()
    return figure


def plot_entity_geometry(data: TrajectoryPlotData) -> Figure:
    """Compare scenario-level bus and generator trajectory geometry."""
    metrics = (
        ("mean_path_length", "Path length"),
        ("mean_net_displacement", "Net displacement"),
        ("mean_path_efficiency", "Path efficiency"),
        (
            "mean_local_directional_consistency",
            "Local consistency",
        ),
        (
            "mean_global_directional_alignment",
            "Global alignment",
        ),
    )
    selected = data.scenarios[
        data.scenarios["entity_type"].isin(("bus", "generator"))
    ]
    figure, axes = plt.subplots(1, len(metrics), figsize=(14.2, 4.8))
    figure.suptitle(
        "Bus versus generator trajectory geometry",
        fontsize=14,
        fontweight="bold",
    )
    for axis, (metric, title), panel in zip(
        axes,
        metrics,
        "abcde",
    ):
        pivot = selected.pivot(
            index=["scenario_id", "cohort_group"],
            columns="entity_type",
            values=metric,
        ).reset_index()
        for row in pivot.itertuples(index=False):
            color = GROUP_COLORS[row.cohort_group]
            axis.plot(
                [0, 1],
                [row.bus, row.generator],
                color=color,
                alpha=0.25,
                linewidth=0.9,
            )
            axis.scatter(
                [0, 1],
                [row.bus, row.generator],
                color=color,
                s=17,
                alpha=0.65,
                zorder=2,
            )
        for group in GROUP_ORDER:
            group_rows = pivot[pivot["cohort_group"] == group]
            axis.plot(
                [0, 1],
                [
                    group_rows["bus"].mean(),
                    group_rows["generator"].mean(),
                ],
                color=GROUP_COLORS[group],
                linewidth=2.4,
                marker="D",
                markersize=4.5,
                zorder=3,
            )
        axis.set_xticks([0, 1], ["Bus", "Generator"])
        axis.set_title(title, fontsize=10)
        axis.set_xlim(-0.25, 1.25)
        _panel_label(axis, panel)
    figure.legend(
        handles=_group_legend_handles(),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.text(
        0.5,
        0.06,
        "Thin lines: scenarios     Thick lines: group means",
        ha="center",
        color="#555555",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.92))
    return figure


def _localization_tables(
    data: TrajectoryPlotData,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    buses = data.entities[data.entities["entity_type"] == "bus"].copy()
    movement = buses.pivot(
        index="entity_id",
        columns="scenario_id",
        values="path_length_share_in_entity_type",
    ).reindex(columns=data.scenario_order)

    pbe = data.bus_predictions.copy()
    pbe["absolute_pbe"] = pbe["PBE"].abs()
    totals = pbe.groupby("scenario_id")["absolute_pbe"].transform("sum")
    if (totals <= 0).any():
        raise ValueError("Scenario PBE totals must be positive")
    pbe["pbe_share"] = pbe["absolute_pbe"] / totals
    pbe_share = pbe.pivot(
        index="entity_id",
        columns="scenario_id",
        values="pbe_share",
    ).reindex(index=movement.index, columns=data.scenario_order)

    cohort = data.cohort.set_index("scenario_id")
    max_pbe_rows = buses.assign(
        is_max_pbe_bus=(
            buses["entity_id"].to_numpy()
            == buses["scenario_id"]
            .map(cohort["max_pbe_bus"])
            .to_numpy()
        )
    )
    max_pbe_ranks = (
        max_pbe_rows[max_pbe_rows["is_max_pbe_bus"]]
        .set_index("scenario_id")[
            ["path_length_rank_in_entity_type", "entity_id"]
        ]
        .reindex(data.scenario_order)
    )
    if max_pbe_ranks.isna().any().any():
        raise ValueError("Unable to locate every maximum-PBE bus")
    return movement, pbe_share, max_pbe_ranks


def _mark_localization_maxima(
    axis: Axes,
    movement: pd.DataFrame,
    pbe_share: pd.DataFrame,
) -> None:
    bus_positions = {
        int(bus): index for index, bus in enumerate(movement.index)
    }
    for column, scenario_id in enumerate(movement.columns):
        movement_bus = int(movement[scenario_id].idxmax())
        pbe_bus = int(pbe_share[scenario_id].idxmax())
        axis.scatter(
            column,
            bus_positions[movement_bus],
            marker="o",
            s=52,
            facecolors="none",
            edgecolors="white",
            linewidths=1.2,
        )
        axis.scatter(
            column,
            bus_positions[pbe_bus],
            marker="x",
            s=42,
            color="white",
            linewidths=1.3,
        )


def plot_localization(data: TrajectoryPlotData) -> Figure:
    """Compare where latent movement and physical PBE are localized."""
    movement, pbe_share, max_pbe_ranks = _localization_tables(data)
    figure = plt.figure(figsize=(14.0, 9.0))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(3.2, 1.0),
        hspace=0.38,
        wspace=0.12,
    )
    movement_axis = figure.add_subplot(grid[0, 0])
    pbe_axis = figure.add_subplot(grid[0, 1], sharey=movement_axis)
    rank_axis = figure.add_subplot(grid[1, :])
    figure.suptitle(
        "Movement and PBE localization",
        fontsize=14,
        fontweight="bold",
    )

    movement_image = movement_axis.imshow(
        100 * movement.to_numpy(dtype=float),
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
    )
    pbe_image = pbe_axis.imshow(
        100 * pbe_share.to_numpy(dtype=float),
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )
    _mark_localization_maxima(movement_axis, movement, pbe_share)
    _mark_localization_maxima(pbe_axis, movement, pbe_share)
    labels = [str(value) for value in data.scenario_order]
    for axis, title in (
        (movement_axis, "Bus share of latent path length (%)"),
        (pbe_axis, "Bus share of absolute PBE (%)"),
    ):
        axis.set_xticks(range(len(labels)), labels, rotation=90)
        axis.set_yticks(
            range(len(movement.index)),
            [str(value) for value in movement.index],
        )
        axis.set_xlabel("Scenario")
        axis.grid(False)
        axis.set_title(title)
    movement_axis.set_ylabel("Bus ID")
    pbe_axis.tick_params(axis="y", labelleft=False)
    figure.colorbar(
        movement_image,
        ax=movement_axis,
        fraction=0.046,
        pad=0.03,
    )
    figure.colorbar(
        pbe_image,
        ax=pbe_axis,
        fraction=0.046,
        pad=0.03,
    )
    _panel_label(movement_axis, "a")
    _panel_label(pbe_axis, "b")

    x = np.arange(len(data.scenario_order))
    ranks = max_pbe_ranks[
        "path_length_rank_in_entity_type"
    ].to_numpy(dtype=float)
    group_by_scenario = data.cohort.set_index("scenario_id")[
        "cohort_group"
    ]
    rank_colors = [
        GROUP_COLORS[group_by_scenario.loc[scenario_id]]
        for scenario_id in data.scenario_order
    ]
    rank_axis.axhspan(0.5, 3.5, color="#2B7A4B", alpha=0.10)
    rank_axis.vlines(
        x,
        ranks,
        len(movement.index) + 0.5,
        color=rank_colors,
        linewidth=3.2,
    )
    rank_axis.scatter(
        x,
        ranks,
        color=rank_colors,
        s=28,
        zorder=3,
    )
    rank_axis.set_xticks(x, labels, rotation=90)
    rank_axis.set_yticks(
        range(
            1,
            int(max(ranks.max(), len(movement.index))) + 1,
            2,
        )
    )
    rank_axis.set(
        xlabel="Scenario",
        ylabel="Movement rank",
        title="Movement rank of the maximum-PBE bus",
    )
    rank_axis.set_ylim(len(movement.index) + 0.5, 0.5)
    rank_axis.text(
        0.99,
        0.90,
        "Green band: top three movement ranks",
        transform=rank_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#45634F",
    )
    _panel_label(rank_axis, "c")
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor="#333333",
                markersize=7,
                label="Maximum movement",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="none",
                color="#333333",
                markersize=7,
                label="Maximum PBE",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
    )
    figure.subplots_adjust(top=0.87, bottom=0.08)
    return figure


PLOTTERS: tuple[
    tuple[str, Callable[[TrajectoryPlotData], Figure]],
    ...,
] = (
    ("metric_intuition", plot_metric_intuition),
    ("cohort_selection", plot_cohort_selection),
    ("group_metric_comparison", plot_group_metric_comparison),
    ("transition_profiles", plot_transition_profiles),
    ("path_length_vs_efficiency", plot_path_efficiency),
    ("bus_vs_generator_geometry", plot_entity_geometry),
    ("movement_pbe_localization", plot_localization),
)


def generate_trajectory_metric_figures(
    spec: TrajectoryPlotSpec,
) -> tuple[Path, ...]:
    """Generate the complete PNG and PDF trajectory figure package."""
    _apply_style()
    data = load_trajectory_plot_data(spec)
    outputs: list[Path] = []
    for stem, plotter in PLOTTERS:
        figure = plotter(data)
        outputs.extend(
            _export_figure(
                figure,
                spec.output_dir,
                stem,
                spec.dpi,
            )
        )
    return tuple(outputs)


def _default_spec(root: Path) -> TrajectoryPlotSpec:
    return TrajectoryPlotSpec(
        scores=root
        / "artifacts/scenario_select/test_scenario_scores.parquet",
        cohort=root
        / "artifacts/scenario_select/selected_test_cohort.parquet",
        entity_metrics=root
        / "artifacts/trajectory_metrics/entity_trajectory_metrics.parquet",
        step_metrics=root
        / "artifacts/trajectory_metrics/step_trajectory_metrics.parquet",
        scenario_summary=root
        / "artifacts/trajectory_metrics/scenario_trajectory_summary.parquet",
        bus_predictions=root
        / (
            "artifacts/trajectory_capture/cohort_instrumented/"
            "case14_ieee_predictions.parquet"
        ),
        output_dir=root / "figure/trajectory_metrics",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone trajectory visualization parser."""
    defaults = _default_spec(Path("."))
    parser = argparse.ArgumentParser(
        description=(
            "Generate reusable latent trajectory metric figures "
            "as PNG and PDF files."
        )
    )
    parser.add_argument("--scores", type=Path, default=defaults.scores)
    parser.add_argument("--cohort", type=Path, default=defaults.cohort)
    parser.add_argument(
        "--entity-metrics",
        type=Path,
        default=defaults.entity_metrics,
    )
    parser.add_argument(
        "--step-metrics",
        type=Path,
        default=defaults.step_metrics,
    )
    parser.add_argument(
        "--scenario-summary",
        type=Path,
        default=defaults.scenario_summary,
    )
    parser.add_argument(
        "--bus-predictions",
        type=Path,
        default=defaults.bus_predictions,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults.output_dir,
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate figures from command-line arguments."""
    args = build_parser().parse_args(argv)
    outputs = generate_trajectory_metric_figures(
        TrajectoryPlotSpec(
            scores=args.scores,
            cohort=args.cohort,
            entity_metrics=args.entity_metrics,
            step_metrics=args.step_metrics,
            scenario_summary=args.scenario_summary,
            bus_predictions=args.bus_predictions,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
