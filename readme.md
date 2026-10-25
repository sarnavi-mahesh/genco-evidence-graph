# GENCO Evidence Graph
GENCO Evidence Graph is a queryable RDF/PROV-O layer connecting GENCO prediction lineage with learned-reference based on embeddings and physics evidence.
It enables SPARQL-based inspection of where predictions came from, how familiar inference scenarios are, and whether their outputs 
are physically consistent.

## Installation
This setup activates the evidence environment so that `genco-evidence` can be called directly. 
Clone and enter the public repository:

```bash
git clone https://github.com/sarnavi-mahesh/genco-evidence-graph.git
cd <genco-evidence-repository-directory>
```
Replace the placeholder with the directory name.

Create amd activate the virtual environment. I strongly recommend using Python 3.12 for the environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

After activation, confirm this:

```bash
which python
python --version
```
The Python path should end with:

```text
<genco-evidence-repository-directory>/.venv/bin/python
```

Install the locked dependencies and genco-evidence-graph package from the current checkout:

```bash
python -m pip install -r requirements-py312.lock
python -m pip install . --no-deps
```

Check dependency consistency and confirm that the installed package imports successfully :

```bash
python -m pip check
python -c \
  "import genco_evidence_graph; print(genco_evidence_graph.__file__)"
```

Confirm that the CLI is available:

```bash
genco-evidence --help
```

This evidence graph is built to support GENCO's published checkpoints and benchmark networks. 
However, this package is tested and currently supports `case14_ieee` with the published seed-42 checkpoint. 

Follow[`reproduction_graphkit.md`](reproduction_graphkit.md) to create the locked graphkit environments and 
download the Hugging Face inputs. Then, from the genco evidence graph repository root, 
set the public graphkit checkout path and proceed to build evidence graphs, validate, and query them.


Select graphkit without activating its separate environment:

```bash
export GRAPHKIT_DIR="/Users/srv/Work/Development/gridfm-graphkit"
export GRAPHKIT_PYTHON="$GRAPHKIT_DIR/.venv/bin/python"
```
Confirm both CLIs:

```bash
genco-evidence --help
"$GRAPHKIT_PYTHON" -m gridfm_graphkit --help
```

The following must already exist:

```bash
test -f configs/case14_seed42_test_embeddings.yaml
test -f configs/baseline_equivalence_tolerances.yaml
test -f configs/graphkit-requirements.lock
test -f configs/graphkit-source.txt
test -f genco-opfdata/case14_ieee/seed42/best_model_state_dict.pt
test -f genco-opfdata/case14_ieee/seed42/normalizer_stats.pt
test -f genco-opfdata/splits/train.pt
test -f genco-opfdata/splits/val.pt
test -f genco-opfdata/splits/test.pt
```
These commands produce no output when all files exist. If there are 
missing files, follow [`reproduction_graphkit.md`](reproduction_graphkit.md) to download 
model checkpoints and opf dataset from Hugging Face and create the 
locked graphkit environments.


## Baseline inference orchestration for test and training scenarios

Important to pay attention before each baseline run:

- the selected `--output-dir` must not exist or must be empty;
- the `--run-name` must be unique such that it uniquely identifies one completed run in 
graphkit's `mlruns`;
- the graphkit checkout must remain clean and at the identified commit.

`run_baseline_inference()` rejects a non-empty output directory.

### Run 1: embedding-enabled test scenarios

This run generates predictions and final embeddings for the 15,000-scenario test split:

```bash
genco-evidence run-baseline \
  --graphkit-python "$GRAPHKIT_PYTHON" \
  --graphkit-workdir "$GRAPHKIT_DIR" \
  --config configs/case14_seed42_test_embeddings.yaml \
  --checkpoint genco-opfdata/case14_ieee/seed42/best_model_state_dict.pt \
  --normalizer genco-opfdata/case14_ieee/seed42/normalizer_stats.pt \
  --data-path data \
  --split-artifact genco-opfdata/splits/test.pt \
  --output-dir artifacts/case14_seed42_test_embeddings \
  --run-name case14_seed42_test_embeddings \
  --graphkit-requirements-lock configs/graphkit-requirements.lock \
  --graphkit-source-record configs/graphkit-source.txt \
  --dataset-split test
```

### Run 2: test scenarios reproduction

This run uses the same input identity but disables embedding 
extraction by passing `--no-embeddings`:

```bash
genco-evidence run-baseline \
  --graphkit-python "$GRAPHKIT_PYTHON" \
  --graphkit-workdir "$GRAPHKIT_DIR" \
  --config configs/case14_seed42_test_embeddings.yaml \
  --checkpoint genco-opfdata/case14_ieee/seed42/best_model_state_dict.pt \
  --normalizer genco-opfdata/case14_ieee/seed42/normalizer_stats.pt \
  --data-path data \
  --split-artifact genco-opfdata/splits/test.pt \
  --output-dir artifacts/case14_seed42_test_no_embeddings \
  --run-name case14_seed42_test_no_embeddings \
  --graphkit-requirements-lock configs/graphkit-requirements.lock \
  --graphkit-source-record configs/graphkit-source.txt \
  --dataset-split test \
  --no-embeddings
```

This run intentionally has no embedding artifact. The goal here to 
make sure the predictions of this run are the same as predictions of
the embedding-enabled run. The comparison is done in the next step.

### Compare runs 1 and 2

```bash
genco-evidence compare-baseline \
  --prediction-parquet artifacts/case14_seed42_test_no_embeddings/case14_ieee_predictions.parquet \
  --generator-prediction-parquet artifacts/case14_seed42_test_no_embeddings/case14_ieee_gen_predictions.parquet \
  --expected-prediction-parquet artifacts/case14_seed42_test_embeddings/case14_ieee_predictions.parquet \
  --expected-generator-prediction-parquet artifacts/case14_seed42_test_embeddings/case14_ieee_gen_predictions.parquet \
  --actual-run-manifest artifacts/case14_seed42_test_no_embeddings/run_manifest.json \
  --expected-run-manifest artifacts/case14_seed42_test_embeddings/run_manifest.json \
  --tolerance-config configs/baseline_equivalence_tolerances.yaml
```

The comparison verifies:

- both manifest identities;
- runtime and lock identity;
- checkpoint, normalizer, configuration and split hashes;
- matching row keys and schemas;
- all mandatory bus and generator fields;
- approved row-level tolerances.

A successful command prints the bus and generator row counts, tolerance-policy identity, and maximum absolute difference for each compared field.

### Prepare the training-reference split view

graphkit predicts its `test.pt` split. To run prediction over the 
training split and get its embeddings, the evidence tool creates 
a swapped split view:

```bash
genco-evidence prepare-reference-split \
  --train-split genco-opfdata/splits/train.pt \
  --validation-split genco-opfdata/splits/val.pt \
  --test-split genco-opfdata/splits/test.pt \
  --output-dir artifacts/case14_train_prediction_splits
```
The output mapping is:

```text
Original test.pt   → generated train.pt
Original val.pt    → generated val.pt
Original train.pt  → generated test.pt
```
The output directory must be empty. `prepare_reference_prediction_splits()` rejects a non-empty directory and verifies every copied file hash.

This causes graphkit's prediction path to process the authoritative training scenarios.

You can verify that the new training reference is now in the test split by checking:

```bash
cmp -s artifacts/case14_train_prediction_splits/test.pt genco-opfdata/splits/train.pt
result=$?

if [ "$result" -eq 0 ]; then
  echo "Files are identical"
elif [ "$result" -eq 1 ]; then
  echo "Files differ"
else
  echo "Comparison failed" >&2
fi
```

### Run 3: embedding-enabled training reference

This run processes the complete 270,000 scenario training reference:

```bash
genco-evidence run-baseline \
  --graphkit-python "$GRAPHKIT_PYTHON" \
  --graphkit-workdir "$GRAPHKIT_DIR" \
  --config configs/case14_seed42_test_embeddings.yaml \
  --checkpoint genco-opfdata/case14_ieee/seed42/best_model_state_dict.pt \
  --normalizer genco-opfdata/case14_ieee/seed42/normalizer_stats.pt \
  --data-path data \
  --split-artifact artifacts/case14_train_prediction_splits/test.pt \
  --output-dir artifacts/case14_seed42_train_embeddings \
  --run-name case14_seed42_train_embeddings \
  --graphkit-requirements-lock configs/graphkit-requirements.lock \
  --graphkit-source-record configs/graphkit-source.txt \
  --dataset-split train
```

`--dataset-split train` labels the generated keyed embeddings as training-reference embeddings even though GraphKit receives them through the generated `test.pt` prediction view.

### What each `run_manifest.json` records under `artifacts/run-name/`

The wrapper automatically records:

- run ID, name, status and timestamps;
- graphkit commit;
- seed, grid, task and split;
- embedding status and dimension;
- Hugging Face dataset revision;
- checkpoint, normalizer and configuration hashes;
- effective configuration hash;
- split and raw-dataset hashes;
- copied MLflow metadata and hash;
- Python/platform/package runtime identity;
- graphkit lock and source-record hashes;
- hashes for all output artifacts.

No user-authored manifest is needed for these three runs.

### Safely reuse a completed embedding run

`--reuse-verified-run` verifies and reuses an already completed 
embedding run; it does not continue an interrupted graphkit 
inference. It can be used to build the evidence graph without running 
inference again to generate predictions and embeddings. The run name must exactly match the value already 
stored in `run_manifest.json`. It does not support the test run without embeddings.

```bash
genco-evidence run-baseline \
  --graphkit-python "$GRAPHKIT_PYTHON" \
  --graphkit-workdir "$GRAPHKIT_DIR" \
  --config configs/case14_seed42_test_embeddings.yaml \
  --checkpoint genco-opfdata/case14_ieee/seed42/best_model_state_dict.pt \
  --normalizer genco-opfdata/case14_ieee/seed42/normalizer_stats.pt \
  --data-path data \
  --split-artifact genco-opfdata/splits/test.pt \
  --output-dir artifacts/case14_seed42_test_embeddings \
  --run-name case14_seed42_coherent_final_retry \
  --graphkit-requirements-lock configs/graphkit-requirements.lock \
  --graphkit-source-record configs/graphkit-source.txt \
  --dataset-split test \
  --reuse-verified-run
```

## Evidence Graph Flow
The checkpoint is downloaded from a pinned repository revision; its path, revision, 
and SHA-256 are recorded, and its bytes are checked against the inference manifest.
Inference lineage is `checkpoint -> inference -> prediction`. Public MLflow
run `1927fe80558a48539ee37be478706781` is an independently authenticated,
co-published, and configuration-compatible training evidence. However, the published 
MLflow run does not record the checkpoint’s hash, and its historical training 
commit is unavailable. Authenticating the metadata only proves which metadata we 
have but it does not prove that this run produced the checkpoint. The complete 
normalized metadata inventory and original upstream training chain see 
[`reproduction_graphkit.md`](reproduction_graphkit.md) are authenticated before any 
training parameter, metric, timestamp, or scenario list value is parsed. 


You can list the test scenario ids through the CLI:
```bash
genco-evidence list-scenarios \
  --split-artifact genco-opfdata/splits/test.pt
```
Use the `--prediction-id` and provide a scenario id to build its evidence graph. The
following commands builds the evidence graph for tests scenario 215429 and 
validates it against the SHACL shapes
```bash
genco-evidence build-evidence \
  --dataset-id opfdata_case14_ieee \
  --checkpoint genco-opfdata/case14_ieee/seed42/best_model_state_dict.pt \
  --prediction-id 215429 \
  --prediction-parquet artifacts/case14_seed42_test_embeddings/case14_ieee_predictions.parquet \
  --generator-prediction-parquet artifacts/case14_seed42_test_embeddings/case14_ieee_gen_predictions.parquet \
  --target-keyed-embeddings artifacts/case14_seed42_test_embeddings/case14_ieee_keyed_embeddings.parquet \
  --reference-keyed-embeddings artifacts/case14_seed42_train_embeddings/case14_ieee_keyed_embeddings.parquet \
  --reference-prediction-parquet artifacts/case14_seed42_train_embeddings/case14_ieee_predictions.parquet \
  --reference-generator-prediction-parquet artifacts/case14_seed42_train_embeddings/case14_ieee_gen_predictions.parquet \
  --reference-run-manifest artifacts/case14_seed42_train_embeddings/run_manifest.json \
  --reference-split-artifact genco-opfdata/splits/train.pt \
  --reference-config-artifact configs/case14_seed42_test_embeddings.yaml \
  --training-run-metadata genco-opfdata/mlflow/train/case14_ieee/seed42/1927fe80558a48539ee37be478706781 \
  --original-upstream-training-metadata-root artifacts/original_upstream_training_metadata \
  --model-repository-path genco-opfdata \
  --raw-dataset-path data/case14_ieee/raw \
  --split-artifact genco-opfdata/splits/test.pt \
  --normalizer-artifact genco-opfdata/case14_ieee/seed42/normalizer_stats.pt \
  --config-artifact configs/case14_seed42_test_embeddings.yaml \
  --run-manifest artifacts/case14_seed42_test_embeddings/run_manifest.json \
  --reproduction-run-manifest artifacts/case14_seed42_test_no_embeddings/run_manifest.json \
  --baseline-tolerance-config configs/baseline_equivalence_tolerances.yaml \
  --expected-dataset-revision 78c3af1e915284e8c7c31f84879878bcf06db6dd \
  --expected-graphkit-commit f104900245cba26d1380037eea562cf633a97484 \
  --output artifacts/case14_scenario_215429_evidence.ttl
  
```

```bash
genco-evidence validate \
  --graph artifacts/case14_scenario_215429_evidence.ttl \
  --shapes ontology/shapes.ttl
```

## RDF namespace

`https://genco.example/` URIs are stable RDF identifiers used by the
evidence graph prototype. They are not browser endpoints and do not require network access.
The graph is inspected locally through RDFLib/SPARQL, and the ontology is
supplied in [`ontology/vocabulary.ttl`](ontology/vocabulary.ttl). You will notice such namespace URI
when you query the evidence graph through the CLI or Python API.

Run the evidence graph queries through named queries. Get the nearest reference scenarios based on embeddings.
```bash
genco-evidence query --graph artifacts/case14_scenario_215429_evidence.ttl --named reference-cases
```
Get residuals and violations for the target prediction.
```bash
genco-evidence query --graph artifacts/case14_scenario_215429_evidence.ttl --named residuals
genco-evidence query --graph artifacts/case14_scenario_215429_evidence.ttl --named violations
```
### Named query legend

Named queries retrieve evidence already stored in the graph; they do not rerun inference or recompute physics. Their definitions are in [`genco_evidence_graph/queries.py`](genco_evidence_graph/queries.py).

| Named query | What it answers | Main output |
|---|---|---|
| `associated-training-evidence` | Which training run is associated with the checkpoint, and what are the limits of that association? | Checkpoint, associated training run, training dataset/split, association status, checkpoint-production verification status, evidence basis, and provenance limitations i.e. authenticated metadata does not prove that the run produced the checkpoint. |
| `checkpoint` | Which checkpoint was used to generate the prediction? | Prediction, checkpoint resource, checkpoint ID, and SHA-256, following `prediction -> inference -> checkpoint`. |
| `reference-cases` | Which reference scenarios are nearest to the target prediction in embedding space? | Reference scenario IDs, distances from the target, and ranks, ordered by rank. These are learned-representation neighbours, not necessarily physically identical scenarios. |
| `embedding-distance` | How far is the target prediction from its nearest reference scenario? | Nearest-reference embedding distance per prediction. Smaller means closer in the embedding representation. |
| `residuals` | How well does the target prediction satisfy the recorded power-balance checks? | Mean and maximum power-balance error (PBE), bus index of maximum PBE, mean active residual in MW, and mean reactive residual in MVAr. Missing optional values appear as blank cells. |
| `violations` | How many recorded operating-limit violations does the target prediction have? | Total violation count and separate voltage, reactive-power, and active-power violation counts. |
| `available-violation-types` | Which violation categories are reported in the graph? | Available violation types, distinguishing reporting coverage from violation counts. An unreported category must not be interpreted as having zero violations. |
| `available-residual-types` | Which residual categories are reported in the graph? | Available residual types. This describes reporting coverage, not residual magnitudes. |
| `reproduction` | What successful reproduction evidence is recorded? | Reproduction and target run IDs, environment SHA-256, platform and package versions, tolerance-policy ID/hash, and maximum voltage and power/residual differences. Only records with `comparisonStatus "passed"` are returned. |
| `high-residual-distance` | Does the target prediction have both high physical error and high embedding distance from reference scenario? | Target prediction, its maximum PBE, and its nearest-reference distance. Both values must be strictly greater than the supplied thresholds. |

`high-residual-distance` requires two thresholds values. `--residual-threshold` for maximum PBE value of the target scenario 
and a second one `--distance-threshold` for its nearest-reference embedding distance. If predictions are both physically inconsistent 
and far from reference scenarios in embedding space?


