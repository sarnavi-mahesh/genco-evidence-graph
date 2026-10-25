from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest
from rdflib import Graph
from rdflib.namespace import RDF

from genco_evidence_graph.cli import build_parser
from genco_evidence_graph.provenance import sha256_file
from genco_evidence_graph.rdf import GENCO
from genco_evidence_graph.scenario_selection_provenance import (
    CohortProvenanceSpec,
    build_cohort_provenance_graph,
    verify_cohort_provenance_inputs,
    write_cohort_provenance,
)
from genco_evidence_graph.validation import validate_evidence_graph


REPO_ROOT = Path(__file__).resolve().parents[1]
COHORT_SHAPES = (
    REPO_ROOT
    / "ontology/scenario_selection_shapes.ttl"
)


def _record(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True)
        + "\n"
    )


def _cohort_records() -> list[dict]:
    records = []
    groups = (
        "familiar_good",
        "familiar_bad",
        "unfamiliar_good",
        "unfamiliar_bad",
    )
    scenario_id = 1
    for group in groups:
        for rank in range(1, 6):
            reason = (
                "corner"
                if rank <= 3
                else (
                    "inner_boundary"
                    if rank == 4
                    else "physical_diversity"
                )
            )
            records.append(
                {
                    "scenario_id": str(scenario_id),
                    "cohort_group": group,
                    "selection_reason": reason,
                    "selection_rank": rank,
                    "nearest_reference_scenario_id": str(
                        100 + scenario_id
                    ),
                    "nearest_reference_distance": (
                        0.1 + scenario_id / 1000
                    ),
                    "mean_absolute_pbe": (
                        0.001 + scenario_id / 100000
                    ),
                    "max_absolute_pbe": (
                        0.002 + scenario_id / 100000
                    ),
                    "max_pbe_bus": scenario_id % 9,
                    "distance_percentile": (
                        scenario_id / 100
                    ),
                    "residual_percentile": (
                        scenario_id / 100
                    ),
                    "max_pbe_percentile": (
                        scenario_id / 100
                    ),
                    "corner_score": (
                        scenario_id / 100
                    ),
                    "boundary_score": (
                        scenario_id / 100
                    ),
                }
            )
            scenario_id += 1
    return records


def _write_fixture(
    tmp_path: Path,
) -> CohortProvenanceSpec:
    scores = tmp_path / "scores.parquet"
    score_manifest = tmp_path / "score_manifest.json"
    summary = tmp_path / "summary.json"
    cohort = tmp_path / "cohort.parquet"
    cohort_manifest = tmp_path / "cohort_manifest.json"
    diagnostics = tmp_path / "diagnostics.json"
    references = tmp_path / "references.csv"
    output = tmp_path / "cohort_provenance.ttl"

    pd.DataFrame(
        {
            "scenario_id": [
                str(index)
                for index in range(1, 21)
            ],
        }
    ).to_parquet(scores, index=False)

    input_records = {}
    for index in range(7):
        source = tmp_path / f"source_{index}.bin"
        source.write_bytes(f"source-{index}".encode())
        input_records[f"source_{index}"] = _record(source)

    _write_json(
        score_manifest,
        {
            "schema_version": 1,
            "artifact_type": "scenario_scoring_table",
            "scoring_policy_id": "genco-scenario-scoring-v1",
            "distance_metric": "euclidean",
            "pooling_method": (
                "mean_bus_concatenate_mean_generator"
            ),
            "residual_metric": "mean_absolute_bus_pbe",
            "numeric_dtype": "float32-neighbor-search",
            "target_scenario_count": 20,
            "reference_scenario_count": 100,
            "entity_embedding_dimension": 4,
            "pooled_embedding_dimension": 8,
            "verified_identity": {
                "checkpoint_sha256": "a" * 64,
                "dataset_revision": "b" * 40,
                "graphkit_commit": "c" * 40,
                "target_run_id": "target-run",
                "reference_run_id": "reference-run",
            },
            "inputs": input_records,
            "output": _record(scores),
        },
    )

    group_counts = {
        "familiar_good": 10,
        "familiar_bad": 11,
        "unfamiliar_good": 12,
        "unfamiliar_bad": 13,
    }
    _write_json(
        summary,
        {
            "schema_version": 1,
            "artifact_type": (
                "scenario_score_distribution_summary"
            ),
            "scoring_policy_id": "genco-scenario-scoring-v1",
            "scenario_count": 20,
            "assigned_scenario_count": 20,
            "excluded_scenario_count": 0,
            "quantile_method": "linear",
            "percentile_policy": {
                "rank_method": "average",
                "low_max": 0.25,
                "high_min": 0.75,
            },
            "spearman_distance_vs_mean_absolute_pbe": 0.5,
            "group_candidate_counts": group_counts,
            "score_table": _record(scores),
            "score_manifest": _record(score_manifest),
        },
    )

    selected = _cohort_records()
    pd.DataFrame(selected).to_parquet(
        cohort,
        index=False,
    )
    references.write_text(
        "scenario_id\n"
        + "\n".join(
            str(index)
            for index in range(101, 121)
        )
        + "\n"
    )

    _write_json(
        diagnostics,
        {
            "schema_version": 1,
            "artifact_type": (
                "contrasting_scenario_selection_diagnostics"
            ),
            "selection_policy_id": (
                "genco-contrasting-cohort-v2"
            ),
            "status": "selected",
            "score_table": _record(scores),
            "score_manifest": _record(score_manifest),
            "distribution_summary": _record(summary),
            "group_candidate_counts": group_counts,
            "selected_scenario_count": 20,
            "selected_training_reference_count": 20,
        },
    )

    _write_json(
        cohort_manifest,
        {
            "schema_version": 1,
            "artifact_type": (
                "frozen_contrasting_scenario_cohort"
            ),
            "scoring_policy_id": "genco-scenario-scoring-v1",
            "selection_policy_id": (
                "genco-contrasting-cohort-v2"
            ),
            "score_table": _record(scores),
            "score_manifest": _record(score_manifest),
            "distribution_summary": _record(summary),
            "diagnostics": _record(diagnostics),
            "cohort_artifact": _record(cohort),
            "training_reference_ids": _record(references),
            "threshold_policy": {
                "low_percentile": 0.25,
                "high_percentile": 0.75,
                "low_distance_max": 0.1,
                "high_distance_min": 0.2,
                "low_residual_max": 0.01,
                "high_residual_min": 0.02,
            },
            "selection_policy": {
                "per_group": 5,
                "corner_representatives": 3,
                "inner_boundary_representatives": 1,
                "physical_diversity_representatives": 1,
                "corner_metric": (
                    "euclidean_percentile_corner_distance"
                ),
                "boundary_metric": (
                    "euclidean_percentile_inner_boundary_distance"
                ),
                "physical_diversity_priority": [
                    "novel_max_pbe_bus",
                    "novel_nearest_reference",
                    "max_pbe_percentile_separation",
                    "corner_score",
                    "numeric_scenario_id",
                ],
            },
            "group_candidate_counts": group_counts,
            "selected_test_scenario_count": 20,
            "selected_training_reference_count": 20,
            "selected_scenarios": selected,
        },
    )

    return CohortProvenanceSpec(
        scores=scores,
        score_manifest=score_manifest,
        summary=summary,
        cohort=cohort,
        cohort_manifest=cohort_manifest,
        diagnostics=diagnostics,
        training_reference_ids=references,
        output=output,
    )


def test_cohort_provenance_verifies_and_validates(
    tmp_path: Path,
) -> None:
    spec = _write_fixture(tmp_path)

    verified = verify_cohort_provenance_inputs(spec)
    result = write_cohort_provenance(spec)

    assert verified.selection_policy_id == (
        "genco-contrasting-cohort-v2"
    )
    assert len(verified.selected_scenarios) == 20
    assert result.selected_scenario_count == 20
    assert result.selected_reference_count == 20
    assert result.output.is_file()

    graph = Graph()
    graph.parse(result.output, format="turtle")
    validate_evidence_graph(graph, COHORT_SHAPES)

    regenerated = replace(
        spec,
        output=tmp_path / "regenerated.ttl",
    )
    write_cohort_provenance(regenerated)
    assert regenerated.output.read_bytes() == (
        result.output.read_bytes()
    )


def test_cohort_provenance_rejects_modified_artifact(
    tmp_path: Path,
) -> None:
    spec = _write_fixture(tmp_path)
    spec.cohort.write_bytes(b"modified")

    with pytest.raises(
        ValueError,
        match="cohort_artifact hash disagrees",
    ):
        verify_cohort_provenance_inputs(spec)


def test_cohort_provenance_rejects_manifest_row_mismatch(
    tmp_path: Path,
) -> None:
    spec = _write_fixture(tmp_path)
    manifest = json.loads(
        spec.cohort_manifest.read_text()
    )
    manifest["selected_scenarios"][0][
        "selection_reason"
    ] = "inner_boundary"
    _write_json(spec.cohort_manifest, manifest)

    with pytest.raises(
        ValueError,
        match="scenario records disagree",
    ):
        verify_cohort_provenance_inputs(spec)


def test_cohort_shapes_reject_missing_membership(
    tmp_path: Path,
) -> None:
    spec = _write_fixture(tmp_path)
    graph, _ = build_cohort_provenance_graph(spec)
    cohort = next(
        graph.subjects(
            RDF.type,
            GENCO.ContrastingScenarioCohort,
        )
    )
    membership = next(
        graph.objects(
            cohort,
            GENCO.hasCohortMembership,
        )
    )
    graph.remove(
        (
            cohort,
            GENCO.hasCohortMembership,
            membership,
        )
    )

    with pytest.raises(
        ValueError,
        match="failed SHACL validation",
    ):
        validate_evidence_graph(
            graph,
            COHORT_SHAPES,
        )


def test_build_cohort_provenance_command_is_registered() -> None:
    args = build_parser().parse_args(
        [
            "build-cohort-provenance",
            "--scores",
            "scores.parquet",
            "--score-manifest",
            "score_manifest.json",
            "--summary",
            "summary.json",
            "--cohort",
            "cohort.parquet",
            "--cohort-manifest",
            "cohort_manifest.json",
            "--diagnostics",
            "diagnostics.json",
            "--training-reference-ids",
            "references.csv",
            "--output",
            "cohort_provenance.ttl",
        ]
    )

    assert args.command == "build-cohort-provenance"
    assert args.handler.__name__ == (
        "_build_cohort_provenance"
    )


def test_build_cohort_provenance_is_public() -> None:
    import genco_evidence_graph as package

    assert "build_cohort_provenance" in package.__all__
    assert callable(package.build_cohort_provenance)
