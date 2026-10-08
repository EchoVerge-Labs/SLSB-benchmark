# Changelog

Scores are only comparable within one protocol version. Every result records `protocol`
and `slsb_version`.

## v0.4.0 · 2026-10-08

**Only speaker identification changed.** Every other task's split, code and settings are
identical to v0.3, so their v0.3 scores carry over unchanged; only `sid` must be re-run.

### Changed: protocol

- **Speaker identification is split by source video.** Per speaker, about 20% of its
  SLCeleb videos (at least one) form the test set and one more the dev set; no test clip
  shares a recording session with training. Under v0.3's clip-level split all 961–999 test
  clips had siblings from the same video in training, so the task partly measured
  recognising the recording: every model scored 0.99+. Frozen XLS-R 300M drops from 0.995
  to 0.874 on unseen videos (seed 0), which leaves room to separate models.
- `make_splits.py` asserts that no video is in two parts and that every speaker appears in
  train, dev and test.

### Added

- `scripts/log_results.py --carry-over`: log a v0.4 run from a re-run of the changed task
  only. The unchanged tasks are copied from the model's `slsb-v0.3` run, and the run
  records where they came from; a changed task can never be carried over.

## v0.3.1 · 2026-10-07

Tooling only; the protocol is unchanged (still v0.3), so v0.3.0 and v0.3.1 scores are
comparable.

- `scripts/log_results.py` logs a finished results folder to MLflow as one run per model in
  the experiment `slsb-<protocol>`, with every task's mean and s.d. as metrics.
- The run summary JSON records the protocol, slsb version, slsb commit, layer norm and the
  `params.yaml` used. The logged `git_commit` is now slsb's own commit, not that of the
  directory slsb was run from.

## v0.3.0 · 2026-10-07

**v0.2 scores are not comparable with v0.3.**

### Changed: protocol

- **Per-layer layer norm.** Every upstream hidden state is layer-normed over its feature
  axis before the learned weighted sum (s3prl's `Featurizer(normalize=True)`), set by
  `upstream.layer_norm` in `params.yaml`. Without it, layers with a large scale dominate
  the mix: `wav2vec2-large-lv60`'s top 3 layers have ~100x the standard deviation of the
  rest, and on `asr_sinhala` its BiLSTM head never trained (test CER 0.80). With layer
  norm the same head reaches CER 0.20 in 12 epochs, and XLS-R 300M improves slightly
  (CER 0.164 -> 0.157). Set `layer_norm: false` to reproduce v0.2.

## v0.2.1 · 2026-10-07

Documentation and licensing only; the protocol, code paths and data are unchanged from
v0.2.0, so v0.2.0 and v0.2.1 scores are directly comparable.

- New README, `docs/` (protocol, tasks, data, results), figures and their source CSVs.
- MIT `LICENSE` and `CITATION.cff`.

## v0.2.0 · 2026-10-07

A new evaluation protocol: new splits, heads and model selection, plus two task families.
**v0.1 scores are not comparable with v0.2.**

### Fixed: leakage in v0.1

- **Speaker verification** embedded clips with the speaker-ID head, which had been trained
  on the same speakers and clips as the trials. It now trains on SLCeleb dev speakers that
  never appear in the trials.
- **`asr_sinhala`'s split** was random: all 600 test clips came from speakers also in
  training. It is now speaker-disjoint, using OpenSLR-52's speaker map.
- **SLCeleb's Sinhala data is duplicated.** The test set is 1,064 recordings copied under
  39 speaker IDs, and the dev set copies the same audio
  ([details](docs/slceleb_data_issues.md)). As a result, speaker ID was 46% copies of
  training clips, and verification training included copies of test audio.
  `asv_sinhala` is excluded, speaker ID keeps only the 40 Tamil speakers, and verification
  trains on Tamil only.
- **Run seeds never changed the split** (it was cached on first use) and the head
  initialisation was unseeded. Splits are now fixed data, and seeds fully seed each head.

### Changed: protocol

- **Fixed train / dev / test splits** for every task (`data/*/split_v2.json`), built by
  `data_prep/make_splits.py`. Splits are speaker- or recording-disjoint where the data
  allows; emotion and small intent tasks use speaker-disjoint 5-fold cross-validation.
- **Automatic leakage checks:** no identical audio may span two parts of a split, and no
  verification training clip may match a trial clip.
- **Model selection on dev:** a learning-rate grid with early stopping, the grid searched
  on the first seed, and the test set scored once.
- **Standard heads:** a 2-layer BiLSTM + CTC for ASR (previously one linear layer), and
  statistics pooling + AM-softmax for verification (chosen on dev EER over an x-vector).
- **Features cached once per task:** the upstream runs once per task, not once per epoch.
  A full run takes about 3 h per model, down from about 8 h.
- **MLflow** records the protocol, version, chosen learning rate, best epoch and dev score.

### Added

- **Speaker diarization** (`sd_sinhala`, `sd_tamil`, SiTa): a clustering pipeline over the
  verification head, with oracle speech regions and DER scored by `pyannote.metrics`.
  Previously an unimplemented stub.
- **Intent classification** (`ic_banking_sinhala`, `ic_banking_tamil`, `ic_health_tamil`).
- **Documentation:** `docs/` (protocol, tasks, data, results), figures and their source
  CSVs, and this changelog.

### Removed from runs

- `asr_omni_sinhala`: one speaker, so no speaker-disjoint split is possible.
- `asv_sinhala`: duplicated SLCeleb audio (above).

## v0.1.0 · 2026-09-19

The first release: frozen upstream, learned weighted sum, and a single linear head per task
trained for a fixed 10 epochs on a train / test split, covering ASR, speaker ID, verification
and Tamil emotion recognition.
