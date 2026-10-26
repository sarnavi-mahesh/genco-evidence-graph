"""Render the submission workflow architecture figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .render import COLORS, _edge_endpoints, _style


WORKFLOW_FIGURE_ID = "genco-evidence-workflow-v1"


def _panel(
    axes: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    color: str,
) -> None:
    axes.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.035,rounding_size=0.12",
            linewidth=1.2,
            edgecolor=COLORS["line"],
            facecolor=COLORS["panel"],
        )
    )
    axes.text(
        x + width / 2,
        y + height - 0.38,
        title,
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        color=color,
    )


def _box(
    axes: Axes,
    center: tuple[float, float],
    width: float,
    height: float,
    title: str,
    color: str,
    *,
    subtitle: str | None = None,
    title_size: float = 9.5,
    subtitle_size: float = 7.0,
    dashed: bool = False,
) -> None:
    x, y = center
    axes.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            linewidth=1.5,
            linestyle=(0, (4, 3)) if dashed else "solid",
            edgecolor=color,
            facecolor=f"{color}14",
        )
    )
    axes.text(
        x,
        y + (0.17 if subtitle else 0),
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        weight="bold",
        color=color,
        linespacing=1.15,
    )
    if subtitle:
        axes.text(
            x,
            y - 0.22,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=COLORS["muted"],
            linespacing=1.35,
        )


def _arrow(
    axes: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dashed: bool = False,
    color: str | None = None,
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    axes.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.35,
            linestyle=(0, (4, 3)) if dashed else "solid",
            color=color or COLORS["gray"],
            shrinkA=0,
            shrinkB=0,
            connectionstyle=connectionstyle,
        )
    )


def _box_arrow(
    axes: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    start_size: tuple[float, float],
    end_size: tuple[float, float],
    *,
    dashed: bool = False,
    color: str | None = None,
) -> None:
    start_point, end_point = _edge_endpoints(
        start,
        end,
        start_size,
        end_size,
        margin=0.04,
    )
    _arrow(
        axes,
        start_point,
        end_point,
        dashed=dashed,
        color=color,
    )


def _save(figure: Figure, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "genco_evidence_workflow.png"
    svg = output_dir / "genco_evidence_workflow.svg"
    metadata = {
        "Creator": "GENCO Evidence Graph",
        "Description": WORKFLOW_FIGURE_ID,
    }
    figure.savefig(
        png,
        dpi=180,
        facecolor="white",
        metadata=metadata,
    )
    figure.savefig(
        svg,
        facecolor="white",
        metadata={**metadata, "Date": None},
    )
    plt.close(figure)
    return png, svg


def render_workflow(output_dir: Path) -> tuple[Path, Path]:
    """Render the implemented evidence workflow and prospective extension."""
    _style()
    plt.rcParams["svg.fonttype"] = "none"
    figure, axes = plt.subplots(figsize=(18, 10))
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axes.set_xlim(0, 18)
    axes.set_ylim(0, 10)
    axes.set_axis_off()

    axes.text(
        0.5,
        9.55,
        "From GENCO prediction to inspectable evidence",
        fontsize=24,
        weight="bold",
        color=COLORS["ink"],
        va="center",
    )
    axes.text(
        0.5,
        9.10,
        "Implemented workflow with a human user as the primary endpoint",
        fontsize=10.5,
        color=COLORS["muted"],
        va="center",
    )

    _panel(axes, 0.15, 2.35, 2.45, 6.05, "Identified inputs", COLORS["teal"])
    _box(
        axes,
        (1.375, 7.28),
        2.02,
        0.88,
        "Published GENCO\ncheckpoint",
        COLORS["blue"],
        title_size=8.6,
    )
    _box(
        axes,
        (1.375, 5.86),
        2.02,
        0.88,
        "Test scenario\nand split",
        COLORS["teal"],
        title_size=8.6,
    )
    _box(
        axes,
        (1.375, 4.42),
        2.02,
        0.98,
        "Pinned Graphkit source\nand configuration",
        COLORS["gray"],
        title_size=8.1,
    )
    axes.text(
        1.375,
        3.18,
        "Model, data, source, and configuration\nidentities are recorded",
        ha="center",
        va="center",
        fontsize=8.0,
        color=COLORS["muted"],
        linespacing=1.35,
    )

    _box(
        axes,
        (4.02, 5.45),
        2.12,
        1.52,
        "GENCO inference",
        COLORS["orange"],
        subtitle="Predictions + final embeddings\n+ physical metrics",
        title_size=10.5,
        subtitle_size=8.0,
    )
    _box(
        axes,
        (6.55, 5.45),
        2.18,
        1.52,
        "GENCO prediction",
        COLORS["purple"],
        subtitle="Stable prediction\nand run identity",
        title_size=10.5,
        subtitle_size=8.0,
    )

    _panel(
        axes,
        7.85,
        2.15,
        4.55,
        6.25,
        "Evidence categories",
        COLORS["purple"],
    )
    _box(
        axes,
        (10.125, 7.25),
        3.28,
        0.92,
        "GENCO Evidence Graph",
        COLORS["purple"],
        subtitle="Linked, validated evidence object",
        title_size=10.4,
        subtitle_size=8.0,
    )
    _box(
        axes,
        (8.98, 5.48),
        1.82,
        1.58,
        "Physical evidence",
        COLORS["green"],
        subtitle="Power-balance residuals\nReported constraint violations",
        title_size=8.5,
        subtitle_size=7.2,
    )
    _box(
        axes,
        (11.27, 5.48),
        1.82,
        1.58,
        "Embedding evidence",
        COLORS["blue"],
        subtitle="Nearest training references\nFinal-embedding distance",
        title_size=8.3,
        subtitle_size=7.2,
    )
    _box(
        axes,
        (10.125, 3.35),
        3.80,
        1.42,
        "Provenance and reproduction",
        COLORS["green"],
        subtitle=(
            "Artifact hashes and run identity\n"
            "Comparison policy and reproduction result\n"
            "Explicit provenance limitations"
        ),
        title_size=8.8,
        subtitle_size=7.0,
    )
    _arrow(axes, (8.98, 6.75), (8.98, 6.31))
    _arrow(axes, (11.27, 6.75), (11.27, 6.31))
    _box_arrow(
        axes,
        (10.125, 7.25),
        (10.125, 3.35),
        (3.28, 0.92),
        (3.80, 1.42),
    )

    _panel(
        axes,
        12.80,
        2.35,
        2.70,
        6.05,
        "Evidence inspection",
        COLORS["blue"],
    )
    axes.text(
        14.15,
        7.40,
        "Operate package",
        ha="center",
        fontsize=8.0,
        weight="bold",
        color=COLORS["muted"],
    )
    _box(
        axes,
        (14.15, 6.76),
        2.20,
        0.72,
        "Command-line interface",
        COLORS["orange"],
        title_size=7.8,
    )
    _box(
        axes,
        (14.15, 5.76),
        2.20,
        0.72,
        "Typed Python API",
        COLORS["purple"],
        title_size=7.8,
    )
    axes.text(
        14.15,
        5.00,
        "Inspect evidence",
        ha="center",
        fontsize=8.0,
        weight="bold",
        color=COLORS["muted"],
    )
    _box(
        axes,
        (14.15, 4.37),
        2.20,
        0.72,
        "Named SPARQL queries",
        COLORS["blue"],
        title_size=7.8,
    )
    _box(
        axes,
        (14.15, 3.36),
        2.20,
        0.72,
        "Source-bound visual reports",
        COLORS["green"],
        title_size=7.4,
    )

    _box(
        axes,
        (16.83, 5.45),
        1.94,
        1.62,
        "Grid planner\nor operator",
        COLORS["teal"],
        subtitle="Inspect evidence, limitations,\nand source identity",
        title_size=10.0,
        subtitle_size=7.5,
    )

    _box_arrow(
        axes,
        (1.375, 5.45),
        (4.02, 5.45),
        (2.45, 6.05),
        (2.12, 1.52),
    )
    _box_arrow(
        axes,
        (4.02, 5.45),
        (6.55, 5.45),
        (2.12, 1.52),
        (2.18, 1.52),
    )
    _arrow(
        axes,
        (6.55, 6.25),
        (8.445, 7.25),
        connectionstyle="angle,angleA=90,angleB=0,rad=0",
    )
    _box_arrow(
        axes,
        (10.125, 5.45),
        (14.15, 5.45),
        (4.55, 6.25),
        (2.70, 6.05),
    )
    _box_arrow(
        axes,
        (14.15, 5.45),
        (16.83, 5.45),
        (2.70, 6.05),
        (1.94, 1.62),
    )

    _box(
        axes,
        (14.65, 1.22),
        4.05,
        1.12,
        "Potential future application",
        COLORS["gray"],
        subtitle="MCP tool + LLM grid co-planner",
        title_size=8.7,
        subtitle_size=8.0,
        dashed=True,
    )
    _arrow(
        axes,
        (14.65, 2.31),
        (14.65, 1.82),
        dashed=True,
        color=COLORS["blue"],
    )

    axes.plot(
        [0.60, 1.18],
        [0.86, 0.86],
        color=COLORS["gray"],
        linewidth=1.35,
    )
    axes.text(
        1.30,
        0.86,
        "Solid line: implemented workflow",
        va="center",
        fontsize=8.0,
        color=COLORS["muted"],
    )
    axes.plot(
        [3.48, 4.06],
        [0.86, 0.86],
        color=COLORS["blue"],
        linewidth=1.35,
        linestyle=(0, (4, 3)),
    )
    axes.text(
        4.18,
        0.86,
        "Dashed line: prospective extension",
        va="center",
        fontsize=8.0,
        color=COLORS["muted"],
    )
    axes.text(
        9.0,
        0.30,
        "Evidence supports inspection. It does not certify operational safety.",
        ha="center",
        va="center",
        fontsize=9.0,
        color=COLORS["muted"],
        weight="bold",
    )

    return _save(figure, output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the GENCO Evidence Graph workflow figure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figure"),
    )
    args = parser.parse_args(argv)
    outputs = render_workflow(args.output_dir)
    print("\n".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
