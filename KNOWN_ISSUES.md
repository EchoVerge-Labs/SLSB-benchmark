# Known issues

## `er_sinhala` — excluded from `data/`

The Sinhala emotion-recognition dataset (`jithara/sinhala-emotional-tts-dataset`)
has known quality issues (its own `metadata.csv` is broken — empty `emotion`
column, bogus 0.04s durations for every row; see
`data_prep/prep_er_sinhala.py`'s docstring). The prep script is kept in
`data_prep/` for when the upstream dataset is fixed, but `data/er_sinhala/`
itself is **not** copied into this repo. `configs/tasks/er.yaml` only lists
`er_tamil`.

## `sd` (speaker diarization) — oracle speech regions, small Tamil test set

Diarization (added in v0.2) takes its speech regions from the reference RTTM,
so DER reflects speaker discrimination only: missed speech comes from
overlapped speech (1.5-3% of SiTa's speech), and false alarm is 0. Scores are
not comparable with systems that detect speech themselves.

SiTa has 60 Sinhala recordings (10 h) but only 14 Tamil ones (2 h), so the
Tamil test set is 5 recordings (0.74 h) and its DER is noisy. One Sinhala test
recording, `YT_49_BS`, has 600 s of audio but reference turns up to 680 s (the
audio looks cut to 10 minutes); turns are clipped to the audio, so only the
600 s it covers are scored. The embedding
head trains on both languages' train recordings, each recording's speakers
being separate classes; a person who appears in two recordings counts as two
classes.

## `asr_omni_sinhala` — excluded from v0.2 (single speaker)

All 111 clips come from one speaker, so no split can be speaker-disjoint: any
score would be a speaker-dependent test, not comparable with `asr_sinhala` /
`asr_tamil`. It is listed in `slsb.tasks.EXCLUDED_TASK_DIRS`, so `slsb run`
skips it; its data stays in `data/` (DVC-tracked, used by v0.1 results).
`data_prep/make_splits.py` still writes a `single_speaker_random` split for it,
in case it is ever re-enabled.

## ASV: few training speakers, and dev doesn't track test closely

ASV trains on 80 Tamil speakers (SUPERB's VoxCeleb1 has ~1,200), which is why
SUPERB's x-vector overfits here and a statistics-pooling head is used instead
(README "Protocol"). Its layer mix is fixed before the head trains, because
every layer's frames for ~45 h of training audio would not fit on disk.

Dev EER (9 held-out speakers) ranks heads less reliably than one would like:
the head chosen on dev (statistics pooling, dev EER 0.123) is not the one with
the lowest test EER among those tried (an x-vector with margin 0.2), so the
choice of ASV head moves the Tamil test EER by ~0.03. More SLCeleb speakers -- or a larger dev set -- would make
ASV selection more trustworthy. Tamil test EER also varies more across seeds
(~±0.02) than the other tasks.

## No benchmark/pre-training contamination check yet

Nothing here verifies that a benchmark test clip was never in an upstream's
pre-training audio. SLCeleb (ASV/SID) is YouTube audio, so a continued-pretraining
corpus built from YouTube could contain the same videos. An audio-fingerprint
check against each pre-training set is still to be built.

## `ic_banking_sinhala` — no speaker ids

The Sinhala banking intent data names files by recording timestamp only, so its
split cannot be speaker-disjoint (`random_stratified_no_speaker_ids`): the same
person can appear in train and test, which flatters its accuracy. Recording
sessions are no substitute -- most hold a single intent. The banking archives'
licence allows academic / research use only, no commercial use.

## Task status summary

| Task family | Status |
|---|---|
| `asr` | validated on `asr_sinhala`, `asr_tamil`; `asr_omni_sinhala` excluded (see above) |
| `sid` | validated, Tamil speakers only (closed set) |
| `asv` | `asv_tamil` validated (speaker-disjoint); `asv_sinhala` excluded — corrupt SLCeleb Sinhala data |
| `er` | validated on `er_tamil` only (speaker-disjoint 5-fold) |
| `sd` | validated, oracle speech regions (see above) |
| `ic` | validated; `ic_banking_sinhala` has no speaker ids (random split, optimistic) |
