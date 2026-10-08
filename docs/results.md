# Results

## Reference run

**Upstream:** XLS-R 300M (`facebook/wav2vec2-xls-r-300m`) after continued pre-training on
200 h of Sinhala and Tamil with normalised audio, in
[Model-Training-Pipeline](https://github.com/EchoVerge-Labs/Model-Training-Pipeline),
checkpoint 9000. **Protocol** v0.4, **seeds** 0, 1, 2, on one NVIDIA GB10 GPU. Nine tasks
were run under v0.3 (slsb 0.3.0); speaker identification, the only task v0.4 changed, was
re-run under v0.4 (slsb 0.4.0). A full run takes about 2 hours.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/fig2-reference-results-dark.png">
  <img alt="Reference results per task: mean of three seeds with their range. Error rates on the left, accuracies on the right." src="assets/fig2-reference-results-light.png">
</picture>

| Task | Language | Metric | Mean ± s.d. (3 seeds) |
|---|---|---|---|
| ASR | Sinhala | WER ↓ | 0.612 ± 0.006 |
| ASR | Sinhala | CER ↓ | 0.156 ± 0.003 |
| ASR | Tamil | WER ↓ | 0.800 ± 0.006 |
| ASR | Tamil | CER ↓ | 0.285 ± 0.006 |
| Emotion recognition | Tamil | Accuracy ↑ | 0.411 ± 0.022 |
| Emotion recognition | Tamil | Macro-F1 ↑ | 0.393 ± 0.031 |
| Speaker identification | Tamil | Accuracy ↑ | 0.876 ± 0.003 |
| Speaker identification | Tamil | Macro-F1 ↑ | 0.856 ± 0.005 |
| Speaker verification | Tamil | EER ↓ | 0.143 ± 0.016 |
| Speaker diarization | Sinhala | DER ↓ | 0.070 ± 0.007 |
| Speaker diarization | Tamil | DER ↓ | 0.026 ± 0.003 |
| Intent · banking | Sinhala | Accuracy ↑ | 0.992 ± 0.002 \* |
| Intent · banking | Sinhala | Macro-F1 ↑ | 0.992 ± 0.002 \* |
| Intent · banking | Tamil | Accuracy ↑ | 0.842 ± 0.017 |
| Intent · banking | Tamil | Macro-F1 ↑ | 0.827 ± 0.018 |
| Intent · health | Tamil | Accuracy ↑ | 0.541 ± 0.000 |
| Intent · health | Tamil | Macro-F1 ↑ | 0.530 ± 0.001 |

\* Optimistic: no speaker information, so the same speakers can be in train and test.

Per-seed values are in [`results/reference_xlsr300m_copt200h_norm_v0.4.csv`](results/reference_xlsr300m_copt200h_norm_v0.4.csv).
The other models are compared on DagsHub, in the `slsb-v0.4` experiment.

### Reading these numbers

- **Diarization DER is almost entirely speaker confusion.** With oracle speech regions,
  missed speech is about 1% (overlap) and false alarm is 0.
- **Intent · banking Sinhala is near ceiling** (0.97–0.99 for every model benchmarked) and
  separates models poorly; speaker identification, near ceiling under v0.3, is no longer
  (0.84–0.89 under v0.4).
- **Run-to-run variation.** ASR's BiLSTM uses non-deterministic GPU kernels, so re-running
  the same seed moves WER by about ±0.01. Differences smaller than that, or than the
  seed s.d., should not be read as real.
- **Scale differs by language.** Absolute WER, CER and accuracy depend on the corpus and
  script, so compare upstreams within a task, not tasks with each other.

## Comparing across protocol versions

| From → to | Comparable? |
|---|---|
| v0.1 → anything later | **No.** v0.1 trained one linear layer for 10 epochs with no dev set, and its splits leaked (Sinhala ASR test speakers in training; verification trained on its own test speakers). |
| v0.2 → v0.3 | **No.** v0.3 layer-normalises every upstream layer, which changes every task's input. |
| v0.3 → v0.4 | **Yes, except speaker identification**, which v0.4 splits by unseen videos. Every other task is identical, which is why v0.4 runs may carry them over (`log_results.py --carry-over`). |

See the [changelog](../CHANGELOG.md) for each version's changes.

## Regenerating the figures and tables

Every number on this page and in the README is read from the versioned CSVs in
[`results/`](results/):

```bash
python docs/scripts/make_figures.py --print        # figures + tables from the CSVs
python docs/scripts/make_figures.py --refresh results/<run> [results/<rerun>] --print
                                                   # rebuild the CSVs from data/ and run outputs
```

`--refresh` takes one or more run directories; a later one replaces an earlier one task by
task. The reference run combines `results/v0.3/xlsr300m_copt200h_norm` (nine tasks) and
`results/v0.4/xlsr300m_copt200h_norm_sid` (speaker identification).
