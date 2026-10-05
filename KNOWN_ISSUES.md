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

## `asr_omni_sinhala` — excluded from v0.2 (single speaker)

All 111 clips come from one speaker, so no split can be speaker-disjoint: any
score would be a speaker-dependent test, not comparable with `asr_sinhala` /
`asr_tamil`. It is listed in `slsb.tasks.EXCLUDED_TASK_DIRS`, so `slsb run`
skips it; its data stays in `data/` (DVC-tracked, used by v0.1 results).
`data_prep/make_splits.py` still writes a `single_speaker_random` split for it,
in case it is ever re-enabled.

## ASV: few training speakers, and dev doesn't track test closely

ASV trains on 125 speakers (SUPERB's VoxCeleb1 has ~1,200), which is why
SUPERB's x-vector overfits here and a statistics-pooling head is used instead
(README "Protocol"). Its layer mix is fixed before the head trains, because
every layer's frames for ~45 h of training audio would not fit on disk.

Dev EER (14 held-out speakers) ranks heads less reliably than one would like:
across the heads tried, test EER on the two trial lists moved in different
directions from dev. More SLCeleb speakers -- or a larger dev set -- would make
ASV selection more trustworthy. Tamil test EER also varies more across seeds
(~±0.02) than the other tasks.

## No benchmark/pre-training contamination check yet

Nothing here verifies that a benchmark test clip was never in an upstream's
pre-training audio. SLCeleb (ASV/SID) is YouTube audio, so a continued-pretraining
corpus built from YouTube could contain the same videos. An audio-fingerprint
check against each pre-training set is still to be built.

## Task status summary

| Task family | Status |
|---|---|
| `asr` | validated on `asr_sinhala`, `asr_tamil`; `asr_omni_sinhala` excluded (see above) |
| `sid` | validated (closed set) |
| `asv` | validated (speaker-disjoint from its trials) |
| `er` | validated on `er_tamil` only (speaker-disjoint 5-fold) |
| `sd` | in progress — data staged, no evaluator |
