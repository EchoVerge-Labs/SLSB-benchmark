# Known issues

## `er_sinhala` — excluded from `data/`

The Sinhala emotion-recognition dataset (`jithara/sinhala-emotional-tts-dataset`)
has known quality issues (its own `metadata.csv` is broken — empty `emotion`
column, bogus 0.04s durations for every row; see
`data_prep/prep_er_sinhala.py`'s docstring). The prep script is kept in
`data_prep/` for when the upstream dataset is fixed, but `data/er_sinhala/`
itself is **not** copied into this repo. `configs/tasks/er.yaml` only lists
`er_tamil`.

## `sd` (speaker diarization) — no DER runner yet

`data/sd_sinhala/` and `data/sd_tamil/` hold paired `<name>.wav` / `<name>.rttm`
files (staged by `data_prep/prep_sita.py`), but there is no diarization
evaluator wired up — this was true in the original Basemodel_Benchmark repo
too (`prep_sita.py`: "Diarization is evaluated with DER, which needs its own
runner (not the linear-probe/ASR runner)"). `slsb.tasks.speaker_diarization.run()`
raises `NotImplementedError` by design; `slsb run --tasks sd` records each
`sd_<lang>` combo as `status: skipped` with that reason rather than fabricating
a DER score. Building the runner (embedding extraction + clustering + DER
scoring, e.g. via `pyannote.metrics`) is unstarted.

## Task status summary

| Task family | Status |
|---|---|
| `asr` | fully validated |
| `sid` | fully validated |
| `asv` | fully validated (depends on `sid` running first) |
| `er` | validated on `er_tamil` only, using the leakage-free split (see `slsb.utils.datasets.load_classification_task_leakage_free`); plain random split is known to leak speaker/sentence identity |
| `sd` | in progress — data staged, no evaluator |

## `configs/defaults.yaml` vs. the current training loop

`configs/defaults.yaml` documents the intended SUPERB-style downstream-head
protocol. The current probe/CTC heads (`slsb/tasks/_common.py`,
`slsb/tasks/asr.py`) implement `frozen_upstream`, `weighted_sum`, and
`pooling: mean` as specified, but are a single linear layer trained with plain
Adam for a fixed epoch count from `params.yaml` — `hidden_dim`, `num_layers`,
`dropout`, `scheduler`, `warmup_ratio`, and `early_stopping_patience` from
`defaults.yaml` are not yet wired into the training loop.
