"""Graphkit plugin that captures GENCO prediction-head input trajectories."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from gridfm_graphkit.io.registries import MODELS_REGISTRY
from gridfm_graphkit.models.gnn_heterogeneous_gns import (
    GNS_heterogeneous,
)
from gridfm_graphkit.tasks.opf_task import (
    OptimalPowerFlowTask,
)


MODEL_TYPE = "GNS_heterogeneous_trajectory_capture"
TRAJECTORY_STAGE = "prediction_head_input"


def _capture_input(
    target: list[torch.Tensor],
) -> Callable:
    def capture(_module, inputs) -> None:
        if len(inputs) != 1:
            raise RuntimeError(
                "Prediction head hook expected exactly one input tensor"
            )
        target.append(
            inputs[0].detach().to(
                device="cpu",
                copy=True,
            )
        )

    return capture


@MODELS_REGISTRY.register(MODEL_TYPE)
class TrajectoryCaptureGNS(GNS_heterogeneous):
    """Capture each layer's bus and generator prediction-head input."""

    def __init__(self, args) -> None:
        super().__init__(args)
        self._captured_trajectory: (
            dict[str, tuple[torch.Tensor, ...]]
            | None
        ) = None

    def forward(
        self,
        batch,
        return_embeddings: bool = False,
    ):
        if self._captured_trajectory is not None:
            raise RuntimeError(
                "A captured trajectory was not consumed before "
                "the next forward pass"
            )

        bus_states: list[torch.Tensor] = []
        generator_states: list[torch.Tensor] = []
        handles = (
            self.mlp_bus.register_forward_pre_hook(
                _capture_input(bus_states)
            ),
            self.mlp_gen.register_forward_pre_hook(
                _capture_input(generator_states)
            ),
        )
        try:
            result = super().forward(
                batch,
                return_embeddings=return_embeddings,
            )
        finally:
            for handle in handles:
                handle.remove()

        if len(bus_states) != self.num_layers:
            raise RuntimeError(
                "Bus trajectory snapshot count disagrees "
                f"with num_layers: {len(bus_states)} != "
                f"{self.num_layers}"
            )
        if len(generator_states) != self.num_layers:
            raise RuntimeError(
                "Generator trajectory snapshot count disagrees "
                f"with num_layers: {len(generator_states)} != "
                f"{self.num_layers}"
            )

        if return_embeddings:
            _, embeddings = result
            for entity_type, states in (
                ("bus", bus_states),
                ("gen", generator_states),
            ):
                final_embedding = (
                    embeddings[entity_type]
                    .detach()
                    .to(device="cpu")
                )
                if not torch.equal(
                    states[-1],
                    final_embedding,
                ):
                    raise RuntimeError(
                        f"Final {entity_type} trajectory snapshot "
                        "disagrees with the exported final embedding"
                    )

        self._captured_trajectory = {
            "bus": tuple(bus_states),
            "gen": tuple(generator_states),
        }
        return result

    def consume_captured_trajectory(
        self,
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        if self._captured_trajectory is None:
            raise RuntimeError(
                "No captured trajectory is available"
            )
        trajectory = self._captured_trajectory
        self._captured_trajectory = None
        return trajectory


def _local_indices(
    batch_index: torch.Tensor,
) -> np.ndarray:
    values = batch_index.detach().to(
        device="cpu",
        dtype=torch.int64,
    )
    counts = torch.bincount(values)
    return np.concatenate(
        [
            np.arange(
                int(count),
                dtype=np.int64,
            )
            for count in counts
        ]
    )


def _trajectory_table(
    snapshots: tuple[torch.Tensor, ...],
    *,
    scenario_ids: np.ndarray,
    entity_ids: np.ndarray,
    entity_column: str,
    bus_ids: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if not snapshots:
        raise RuntimeError(
            "Trajectory snapshots cannot be empty"
        )
    entity_count = len(scenario_ids)
    dimension = snapshots[0].shape[1]
    if any(
        snapshot.ndim != 2
        or snapshot.shape != (
            entity_count,
            dimension,
        )
        for snapshot in snapshots
    ):
        raise RuntimeError(
            "Trajectory snapshots have inconsistent shapes"
        )

    values = torch.cat(
        snapshots,
        dim=0,
    ).numpy()
    table = {
        "scenario": np.tile(
            scenario_ids,
            len(snapshots),
        ),
        entity_column: np.tile(
            entity_ids,
            len(snapshots),
        ),
        "correction_step": np.repeat(
            np.arange(
                1,
                len(snapshots) + 1,
                dtype=np.int16,
            ),
            entity_count,
        ),
        "embedding_stage": np.repeat(
            TRAJECTORY_STAGE,
            entity_count * len(snapshots),
        ),
    }
    if bus_ids is not None:
        table["bus"] = np.tile(
            bus_ids,
            len(snapshots),
        )
    for index in range(dimension):
        table[f"emb_{index:03d}"] = values[:, index]
    return table


_original_predict_step = (
    OptimalPowerFlowTask.predict_step
)


def _trajectory_predict_step(
    self,
    batch,
    batch_idx,
    dataloader_idx=0,
):
    tables = _original_predict_step(
        self,
        batch,
        batch_idx,
        dataloader_idx,
    )
    if not isinstance(
        self.model,
        TrajectoryCaptureGNS,
    ):
        return tables

    trajectory = (
        self.model.consume_captured_trajectory()
    )
    bus_batch = batch.batch_dict["bus"]
    generator_batch = batch.batch_dict["gen"]
    bus_scenario_ids = (
        batch["scenario_id"][bus_batch]
        .detach()
        .cpu()
        .numpy()
    )
    generator_scenario_ids = (
        batch["scenario_id"][generator_batch]
        .detach()
        .cpu()
        .numpy()
    )
    local_bus_ids = _local_indices(bus_batch)
    local_generator_ids = _local_indices(
        generator_batch
    )
    _, generator_to_bus = batch.edge_index_dict[
        ("gen", "connected_to", "bus")
    ]
    generator_bus_ids = local_bus_ids[
        generator_to_bus.detach().cpu().numpy()
    ]

    tables["bus_trajectory"] = _trajectory_table(
        trajectory["bus"],
        scenario_ids=bus_scenario_ids,
        entity_ids=local_bus_ids,
        entity_column="bus",
    )
    tables["gen_trajectory"] = _trajectory_table(
        trajectory["gen"],
        scenario_ids=generator_scenario_ids,
        entity_ids=local_generator_ids,
        entity_column="idx",
        bus_ids=generator_bus_ids,
    )
    return tables


_trajectory_predict_step._genco_trajectory_capture = True
if not getattr(
    OptimalPowerFlowTask.predict_step,
    "_genco_trajectory_capture",
    False,
):
    OptimalPowerFlowTask.predict_step = (
        _trajectory_predict_step
    )
