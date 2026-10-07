# SLSB-benchmark — Sinhala / Lankan-Tamil Speech Benchmark

A frozen-upstream, SUPERB-style benchmark for comparing self-supervised speech
encoders on Sinhala and Sri Lankan Tamil tasks: the upstream encoder is always
kept frozen, and only a learned weighted sum over its layers plus a standard
per-task downstream head is trained, so results reflect what the upstream's
representations capture.

## Installation

```bash
pip install -e .
# or
pip install git+https://github.com/EchoVerge-Labs/SLSB-benchmark.git@v0.2.0
```

For development (linting + tests):

```bash
pip install -e ".[dev]"
```

> **ARM64 / DGX Spark (GB10) note:** install `torch` + `torchaudio` as a
> matched pair from native aarch64 CUDA wheels. Do **not** install or
> reference `flash-attn` — it does not build on this hardware.

## Usage

```bash
slsb run --upstream facebook/wav2vec2-xls-r-300m \
         --tasks asr,sid,er,sd \
         --data-dir ./data \
         --seeds 0,1,2 \
         --out results/ \
         --mlflow-uri https://dagshub.com/EchoVerge-Labs/SLSB-benchmark.mlflow
```

`--upstream` accepts any Hugging Face repo id, or one of the short aliases
`xlsr`, `mhubert147`, `wavlm_large`. Results are written to `<out>/benchmark_table.csv`
(append-only score table) and `<out>/results_<upstream>.json` (structured
summary of this run). MLflow logging is skipped unless `--mlflow-uri` is set
and `DAGSHUB_TOKEN` is exported.

See `make benchmark` for a minimal example.

## Available tasks

| Family | Data dir(s) | Language(s) | Kind | Primary metric | Status |
|---|---|---|---|---|---|
| `asr` | `asr_sinhala`, `asr_tamil` | Sinhala, Tamil | ASR (CTC) | WER (+ CER) | validated |
| `sid` | `sid` | multilingual | classification | accuracy | validated |
| `asv` | `asv` | Sinhala, Tamil (trial pairs) | verification | EER | validated |
| `er` | `er_tamil` | Tamil | classification | accuracy | validated (speaker-disjoint 5-fold) |
| `sd` | `sd_sinhala`, `sd_tamil` | Sinhala, Tamil | diarization | DER | validated (oracle speech regions) |

Not run from v0.2: `asr_omni_sinhala` (one speaker) and `asv_sinhala`
(SLCeleb's Sinhala data is 1,064 recordings copied under 39 speaker ids, see
[`docs/slceleb_data_issues.md`](docs/slceleb_data_issues.md)); SID is
therefore Tamil-only. `er_sinhala` is **excluded** from this repo's data
pending dataset-quality fixes upstream. See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

## Protocol (v0.2)

Scores from different protocol versions are not comparable; each MLflow run
records `protocol` and `slsb_version`.

| Task | Downstream head | Split (`data/<task>/split_v2.json`) | Selected on |
|---|---|---|---|
| ASR | weighted sum + 2-layer BiLSTM (1024/direction) + CTC over characters, greedy decoding | speaker-disjoint train/dev/test, 70/10/20 | dev CER |
| SID | weighted sum + mean-pool + linear | closed set (every speaker is a class), stratified train/dev/test | dev accuracy |
| ER | weighted sum + mean-pool + linear | speaker-disjoint 5-fold cross-validation; per fold, the next fold is dev; mean over folds | dev accuracy |
| ASV | weighted sum + mean/std statistics pooling + linear embedding, AM-softmax, cosine scoring | trained on 80 SLCeleb Tamil dev speakers, selected on dev trials from 9 more; tested on the Tamil trial list, whose speakers it never sees | dev EER |
| SD | oracle speech regions; speaker-embedding head as for ASV, trained on the train recordings; windows clustered per recording (AHC, cosine), number of speakers never given | recording-disjoint train/dev/test, 40/20/40 | dev DER (threshold + early stopping) |

- **Model selection:** every head is trained per learning rate in `params.yaml`'s
  `lr_grid`, with early stopping on the dev split. The grid is searched on the
  first run seed; later seeds reuse the learning rate it picked. The test split
  is scored once, at the end.
- **Seeds** vary the head's initialisation and batch order. The splits are fixed
  data, the same for every seed and every upstream.
- **Features are extracted once per task.** The upstream is frozen, so its
  hidden states are cached (`<out>/.features/`, deleted when the task finishes)
  and every epoch, learning rate and seed trains on them.
- **ASV departs from SUPERB twice.** (1) Its head is statistics pooling, not
  SUPERB's x-vector, chosen on dev EER (never on test) on the 80-speaker
  Tamil training set: statistics pooling reached dev EER 0.123 +- 0.003 (3
  seeds) against 0.142-0.186 for four x-vector variants, and keeps improving
  for ~13 epochs where the x-vectors peak within 3-6. (2) Storing
  every layer's frames for ~45 h of training audio is too large, so the layer
  weights are fixed first, from a mean-pool speaker classifier on a subset of
  the training speakers.
- `asr_omni_sinhala` is excluded: it has a single speaker, so it can't be
  split speaker-disjointly (its data stays in `data/`; see KNOWN_ISSUES.md).

### What changed from v0.1

v0.1 trained one linear layer for a fixed 10 epochs with no dev set, and:
- `asv` trained its embedding (SID's head) on the same 79 speakers and clips its
  trials test;
- `asr_sinhala`'s split was random: all 600 test clips came from speakers also
  in training;
- `er_tamil` was scored on a single split of 936 clips;
- a task's split was cached on first use, so run seeds never changed it.

### Building the v0.2 data

On top of the v0.1 data:

```bash
python data_prep/prep_asr_sinhala_speakers.py   # data/asr_sinhala/speakers.csv
python data_prep/prep_slceleb_train.py          # data/asv/train_audio/ + train_labels.csv
python data_prep/make_splits.py                 # data/*/split_v2.json
```

## Dataset

`data/` is DVC-tracked (remote: DagsHub). After cloning:

```bash
dvc pull
```

## Project layout

```
src/slsb/        installable package (cli, runner, tasks/, upstream/, metrics/, utils/)
configs/         defaults.yaml + per-task-family configs/tasks/*.yaml
data_prep/       scripts that built data/ from upstream sources (see each script's docstring)
tests/           smoke + protocol tests (no GPU, data or network needed)
```
