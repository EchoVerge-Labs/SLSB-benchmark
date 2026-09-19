# SLSB-benchmark — Sinhala / Lankan-Tamil Speech Benchmark

A frozen-upstream, SUPERB-style benchmark for comparing self-supervised speech
encoders on Sinhala and Sri Lankan Tamil tasks: the upstream encoder is always
kept frozen, and only a lightweight learned-weighted-sum + linear head is
trained per task, so results isolate what the upstream's representations
actually capture rather than how well a big downstream model can compensate.

## Installation

```bash
pip install -e .
# or
pip install git+https://github.com/EchoVerge-Labs/SLSB-benchmark.git@v0.1.0
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
| `asr` | `asr_sinhala`, `asr_tamil`, `asr_omni_sinhala` | Sinhala, Tamil | ASR (CTC) | WER | validated |
| `sid` | `sid` | multilingual | classification | accuracy | validated |
| `asv` | `asv` | Sinhala, Tamil (trial pairs) | verification | EER | validated (depends on `sid`) |
| `er` | `er_tamil` | Tamil | classification | accuracy | validated (leakage-free split) |
| `sd` | `sd_sinhala`, `sd_tamil` | Sinhala, Tamil | diarization | DER | **not implemented** — see below |

`er_sinhala` is **excluded** from this repo's data pending dataset-quality
fixes upstream. `sd_*` data is staged (wav + RTTM pairs) but has no DER
evaluator yet; `slsb run --tasks sd` reports each combo as `skipped` rather
than fabricating a score. See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) for detail
on both.

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
tests/           smoke tests (import + config validation, no GPU/data needed)
```
