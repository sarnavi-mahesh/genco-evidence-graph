# Required graphkit installation and Hugging Face downloads

The standard `pip install gridfm-graphkit` installation is not sufficient 
for evidence reproduction. The evidence workflow requires the pinned public 
graphkit checkout, a dedicated virtual environment inside that checkout, 
the project-supplied dependency lock, and an editable installation of 
the pinned source.

Keep the genco evidence graph environment activated and address this separate 
graphkit environment through `.venv/bin/python` explicitly. Do not activate both environments.
The commands below use explicit virtual-environment executable paths.

Clone the graphkit repo and check out the pinned commit(https://huggingface.co/gridfm/genco-opfdata):

```bash
git clone https://github.com/gridfm/gridfm-graphkit.git
cd gridfm-graphkit
git checkout f104900245cba26d1380037eea562cf633a97484
```
Confirm the source identity:

```bash
git remote get-url origin
git rev-parse HEAD
```

The output must identify:

```text
repository=https://github.com/gridfm/gridfm-graphkit.git
commit=f104900245cba26d1380037eea562cf633a97484
```

### Create the dedicated graphKit environment (from the graphkit directory). 
Use recommended Python 3.12 (https://github.com/gridfm/gridfm-graphkit/blob/main/README.md) for the environment:

```bash
python -m venv .venv
```

### Install the validated Torch stack

```bash
.venv/bin/python -m pip install \
  "torch==2.11.0" \
  "torchvision==0.26.0" \
  "torchaudio==2.11.0"
```

Build and install `torch-scatter` against the installed Torch environment:

```bash
CXXFLAGS="-faligned-allocation" \
.venv/bin/python -m pip install \
--no-build-isolation \
--no-cache-dir \
"torch-scatter==2.1.2"
```
You can also use pre-compiled wheels for `torch-scatter`. 
For example, CPU-only Torch 2.11:
```bash
.venv/bin/python -m pip install \
  --only-binary=:all: \
  "torch-scatter==2.1.2+pt211cpu" \
  -f "https://data.pyg.org/whl/torch-2.11.0+cpu.html"
```

On other operating systems, install the same Torch and `torch-scatter` versions using the matching official Torch/PyG installation method before continuing.

## Install the project-locked graphkit environment

Replace `/absolute/path/to/genco-evidence-graph` with the genco-evidence-graph directory path:

```bash
.venv/bin/python -m pip install \
  -r /absolute/path/to/genco-evidence-graph/configs/graphkit-requirements.lock
```

The new user must use this project supplied lock rather than generating a 
replacement unless necessary (more below). The platform-sensitive Torch packages installed above already satisfy their entries in the lock and should not be reinstalled.

Install graphKit from the pinned checkout without resolving dependencies again:

```bash
.venv/bin/python -m pip install -e . --no-deps
```

### Why graphkit must be installed with `-e` and `--no-deps`

For this evidence workflow, graphkit's dependencies and graphkit's source identity are authenticated separately:

- `genco-evidence-graph/configs/graphkit-requirements.lock` identifies the complete installed dependency environment.
- `genco-evidence-graph/configs/graphkit-source.txt` identifies the graphkit repository and exact Git commit.

Installing graphkit with `-e` causes it to be deliberately omitted from that 
dependency freeze. The `--no-deps` option prevents the editable source installation from resolving 
or changing packages already installed from the approved dependency lock.
The evidence workflow then verifies the GraphKit source separately by checking 
the checkout's Git remote and commit against `genco-evidence-graph/configs/graphkit-source.txt`.

This design cleanly separates two kinds of provenance: graphkit's dependency 
environment and graphkit's source code. 
It also avoids relying only on the package version `0.9.0.3`, because a version number alone does not identify a specific source commit.


Do not modify the pinned GraphKit checkout after installation. The checkout must remain at:

```text
repository=https://github.com/gridfm/gridfm-graphkit.git
commit=f104900245cba26d1380037eea562cf633a97484
```

#### Clean-checkout limitation

Since editable installation executes code from that checkout, do not modify it.
Check that the checkout is clean before running the evidence workflow (from Graphkit directory):

```bash
git -C "$(pwd)" status --porcelain
```

The command must produce no output. Since the current evidence graph package cannot enforce
this clean-tree check, preserving and checking the unmodified checkout is a required user responsibility.

### Verify the environment
Replace `/absolute/path/to/genco-evidence-graph` with the genco-evidence-graph directory path:

```bash
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze --all --exclude-editable \
  > /tmp/graphkit-environment.txt
cmp \
  /tmp/graphkit-environment.txt \
  /absolute/path/to/genco-evidence-graph/configs/graphkit-requirements.lock
git remote get-url origin
git rev-parse HEAD
```

`cmp` typically completes without any output. Any output implies a difference between selected 
graphkit environment and genco evidence graph supplied lock. This usually occurs if 
platform-specific dependencies are installed (e.g. pywin32) or pre-compiled torch-scatter wheels are used.
Verify the differences are genuine and the user may generate a new platform specific lock:

```bash
cp /absolute/path/to/genco-evidence-graph/configs/graphkit-requirements.lock /absolute/path/to/genco-evidence-graph/configs/graphkit-requirements-backup.lock
.venv/bin/python -m pip freeze --all --exclude-editable > /absolute/path/to/genco-evidence-graph/configs/graphkit-requirements.lock
```


### Hugging Face Downloads 

Run these commands from the `genco-evidence-graph` directory root with its environment activated:

```bash
source .venv/bin/activate
```

Confirm the genco-evidence-graph CLI is available:

```bash
genco-evidence --help
```

Install the Hugging Face CLI and confirm it:

```bash
python -m pip install "huggingface_hub[cli]"
hf --help
```

### Download the trained GENCO artifacts

The model artifacts come from:

```text
Repository: gridfm/genco-opfdata
Revision:   74e19e1e210a175aa17023f4db161f0ae9f753cb
```

### Confirm the revision for seed-42 checkpoint is available

Do not rely only on `git ls-remote HEAD`, because `HEAD` may 
have advanced. Query the pinned historical revision from your
terminal:

```bash
python - <<'PY'
from huggingface_hub import HfApi

revision = "74e19e1e210a175aa17023f4db161f0ae9f753cb"

model = HfApi().model_info(
    "gridfm/genco-opfdata",
    revision=revision,
)

if model.sha != revision:
    raise SystemExit(
        f"Resolved model revision {model.sha!r}, expected {revision!r}"
    )

print("Confirmed model revision:", model.sha)
PY
```

Expected output:

```text
Confirmed model revision: 74e19e1e210a175aa17023f4db161f0ae9f753cb
```

### Download the seed-42 checkpoint and normalizer

```bash
hf download gridfm/genco-opfdata \
  --revision 74e19e1e210a175aa17023f4db161f0ae9f753cb \
  --include "case14_ieee/seed42/**" \
  --local-dir genco-opfdata
```
This supplies seed-42 files.


### Download the data splits

```bash
hf download gridfm/genco-opfdata \
  --revision 74e19e1e210a175aa17023f4db161f0ae9f753cb \
  --include "splits/*" \
  --local-dir genco-opfdata
```
This supplies the train/validation/test split pytorch artifacts.

### Confirm the required files

```bash
test -f genco-opfdata/case14_ieee/seed42/best_model_state_dict.pt
test -f genco-opfdata/case14_ieee/seed42/normalizer_stats.pt
test -f genco-opfdata/splits/train.pt
test -f genco-opfdata/splits/val.pt
test -f genco-opfdata/splits/test.pt
```
These commands produce no output when all files exist. Make
sure you have all the above files.

### Download the exact case-14 scenario corpus

The complete dataset comes from:

```text
Repository: gridfm/opfdata_case14_ieee
Revision:   78c3af1e915284e8c7c31f84879878bcf06db6dd
```

This is the full published OPF data corpus for case IEEE14.


### Confirm the revision for data corpus is available

```bash
python - <<'PY'
from huggingface_hub import HfApi

revision = "78c3af1e915284e8c7c31f84879878bcf06db6dd"

dataset = HfApi().dataset_info(
    "gridfm/opfdata_case14_ieee",
    revision=revision,
)

if dataset.sha != revision:
    raise SystemExit(
        f"Resolved dataset revision {dataset.sha!r}, expected {revision!r}"
    )

print("Confirmed dataset revision:", dataset.sha)
PY
```

Expected output:
```text
Confirmed dataset revision: 78c3af1e915284e8c7c31f84879878bcf06db6dd
```

### Download the complete dataset

```bash
hf download gridfm/opfdata_case14_ieee \
  --repo-type dataset \
  --revision 78c3af1e915284e8c7c31f84879878bcf06db6dd \
  --local-dir data/case14_ieee/raw
```

### Prepare the authenticated training-run metadata:
Get the upstream training metadata for the seed-42 run for case ieee14 
from the Hugging Face repository. 
```bash
 hf download gridfm/genco-opfdata \
  --revision 74e19e1e210a175aa17023f4db161f0ae9f753cb \
  --include "mlflow/train/case14_ieee/seed42/1927fe80558a48539ee37be478706781/**" \
  --local-dir artifacts/original_upstream_training_metadata
```

Then generate and verify the portable run tree using the locally preserved Hugging Face revision and per-file cache metadata:

```bash
genco-evidence prepare-training-metadata \
  --original-upstream-root artifacts/original_upstream_training_metadata \
  --model-repository-path genco-opfdata \
  --run-relative-path mlflow/train/case14_ieee/seed42/1927fe80558a48539ee37be478706781 \
  --output-run-path genco-opfdata/mlflow/train/case14_ieee/seed42/1927fe80558a48539ee37be478706781
```

Do not edit either the untouched downloaded metadata tree or the 
generated normalized metadata tree after generation. The evidence 
graph authenticates the source cache records and deterministic 
transforms against both trees.

