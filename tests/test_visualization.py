import json
from pathlib import Path

import pytest
from rdflib import Graph

from genco_evidence_graph.cli import build_parser
from genco_evidence_graph.provenance import sha256_file
from genco_evidence_graph.rdf import GENCO
from genco_evidence_graph.visualization.model import (
    EvidenceShowcaseSpec,
    load_showcase_view,
)


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "artifacts/case14_scenario_215429_evidence.ttl"
BUS_PREDICTIONS = (
    ROOT
    / "artifacts/case14_seed42_test_embeddings"
    / "case14_ieee_predictions.parquet"
)
GENERATOR_PREDICTIONS = (
    ROOT
    / "artifacts/case14_seed42_test_embeddings"
    / "case14_ieee_gen_predictions.parquet"
)
CHECKPOINT = (
    ROOT
    / "genco-opfdata/case14_ieee/seed42"
    / "best_model_state_dict.pt"
)


def _require_real_artifacts() -> None:
    missing = [
        path
        for path in (
            GRAPH,
            BUS_PREDICTIONS,
            GENERATOR_PREDICTIONS,
            CHECKPOINT,
        )
        if not path.is_file()
    ]
    if missing:
        pytest.skip(
            "Visualization integration artifacts are unavailable: "
            + ", ".join(str(path) for path in missing)
        )


def _spec(output_dir: Path, graph: Path = GRAPH) -> EvidenceShowcaseSpec:
    return EvidenceShowcaseSpec(
        graph=graph,
        shapes=ROOT / "ontology/shapes.ttl",
        prediction_parquet=BUS_PREDICTIONS,
        generator_prediction_parquet=GENERATOR_PREDICTIONS,
        checkpoint=CHECKPOINT,
        tolerance_config=(
            ROOT / "configs/baseline_equivalence_tolerances.yaml"
        ),
        output_dir=output_dir,
        project_root=ROOT,
    )


def test_workflow_figure_is_high_resolution_and_claim_bounded(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib.image as mpimg

    from genco_evidence_graph.visualization.workflow import render_workflow

    png, svg = render_workflow(tmp_path)

    assert png.is_file()
    assert svg.is_file()
    image = mpimg.imread(png)
    assert image.shape[1] >= 1_800
    assert image.shape[0] >= 1_000
    svg_text = svg.read_text()
    for label in (
        "From GENCO prediction to inspectable evidence",
        "GENCO Evidence Graph",
        "Evidence inspection",
        "Grid planner",
        "Potential future application",
        "MCP tool + LLM grid co-planner",
        "Evidence supports inspection. It does not certify operational safety.",
    ):
        assert label in svg_text
    assert "stroke-dasharray" in svg_text
    assert "GencoLens" not in svg_text
    assert "\N{EM DASH}" not in svg_text


def test_visualization_cli_commands_parse_without_render_import():
    parser = build_parser()
    for command in (
        "render-evidence-summary",
        "render-evidence-graph",
        "render-evidence-charts",
        "render-evidence-showcase",
    ):
        args = parser.parse_args([command])
        assert args.graph.endswith(".ttl")
        assert args.output_dir == "figure/scenario_215429"


def test_graph_edges_are_clipped_outside_node_boxes():
    pytest.importorskip("matplotlib")
    from genco_evidence_graph.visualization.render import (
        _edge_endpoints,
    )

    start, end = _edge_endpoints(
        (0.0, 0.0),
        (4.0, 0.0),
        (2.0, 1.0),
        (2.0, 1.0),
        margin=0.1,
    )

    assert start == pytest.approx((1.1, 0.0))
    assert end == pytest.approx((2.9, 0.0))


@pytest.mark.integration
def test_load_showcase_view_verifies_canonical_values(tmp_path):
    _require_real_artifacts()
    view = load_showcase_view(_spec(tmp_path))

    assert view.scenario_id == "215429"
    assert view.references[0].scenario_id == "13423"
    assert view.nearest_reference_distance == pytest.approx(0.1574388)
    assert view.mean_pbe == pytest.approx(0.0007663918)
    assert view.max_pbe == pytest.approx(0.001922636)
    assert view.max_pbe_bus == 0
    assert view.total_violations == 1
    assert view.reactive_violations == 1
    assert view.reproduction_passed
    assert view.checkpoint_verified
    assert view.shacl_validated
    assert not view.training_production_identity_verified
    assert len(view.bus_predictions) == 14
    assert len(view.generator_predictions) == 4


@pytest.mark.integration
def test_load_showcase_view_rejects_recorded_hash_mismatch(tmp_path):
    _require_real_artifacts()
    graph = Graph()
    graph.parse(GRAPH, format="turtle")
    artifact = next(
        subject
        for subject, path in graph.subject_objects(GENCO.artifactPath)
        if str(path).endswith("case14_ieee_predictions.parquet")
    )
    graph.remove((artifact, GENCO.sha256, None))
    from rdflib import Literal

    graph.add((artifact, GENCO.sha256, Literal("0" * 64)))
    altered = tmp_path / "hash_mismatch.ttl"
    graph.serialize(altered, format="turtle")

    with pytest.raises(ValueError, match="hash disagrees"):
        load_showcase_view(_spec(tmp_path / "output", altered))


@pytest.mark.integration
def test_load_showcase_view_rejects_shacl_failure(tmp_path):
    _require_real_artifacts()
    graph = Graph()
    graph.parse(GRAPH, format="turtle")
    prediction = next(graph.subjects(None, GENCO.Prediction))
    graph.remove((prediction, GENCO.predictionId, None))
    altered = tmp_path / "invalid.ttl"
    graph.serialize(altered, format="turtle")

    with pytest.raises(ValueError, match="SHACL validation"):
        load_showcase_view(_spec(tmp_path / "output", altered))


@pytest.mark.integration
def test_render_showcase_writes_bound_figure_bundle(tmp_path):
    pytest.importorskip("matplotlib")
    _require_real_artifacts()
    from genco_evidence_graph.visualization.render import render_showcase

    result = render_showcase(_spec(tmp_path / "figures"))

    assert len(result.outputs) == 14
    assert all(path.stat().st_size > 1_000 for path in result.outputs)
    assert result.manifest is not None
    manifest = json.loads(result.manifest.read_text())
    assert manifest["scenario_id"] == "215429"
    assert manifest["presentation_only"] is True
    assert manifest["verification"] == {
        "checkpoint_verified": True,
        "prediction_artifacts_verified": True,
        "shacl_validated": True,
    }
    graph_svg = (tmp_path / "figures" / "evidence_graph.svg").read_text()
    assert "Bus prediction artifact" in graph_svg
    assert "Generator prediction artifact" in graph_svg
    assert "3b69e1e2a6" in graph_svg
    assert "13fce1ba65" in graph_svg
    assert "Bus artifact:" in graph_svg
    assert "Generator artifact:" in graph_svg
    for item in manifest["figures"]:
        path = ROOT / item["path"]
        assert item["sha256"] == sha256_file(path)
