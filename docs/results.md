# Results

## Reference run

**Upstream:** XLS-R 300M (`facebook/wav2vec2-xls-r-300m`) after continued pre-training on
200 h of Sinhala and Tamil with normalised audio, in
[Model-Training-Pipeline](https://github.com/EchoVerge-Labs/Model-Training-Pipeline),
checkpoint 9000. **Protocol** v0.2 (slsb 0.2.0), **seeds** 0, 1, 2, on one NVIDIA GB10 GPU.
The run took about 3 hours.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/fig2-reference-results-dark.png">
  <img alt="Reference results per task: mean of three seeds with their range. Error rates on the left, accuracies on the right." src="assets/fig2-reference-results-light.png">
</picture>

| Task | Language | Metric | Mean ± s.d. (3 seeds) |
|---|---|---|---|
| ASR | Sinhala | WER ↓ | 0.635 ± 0.008 |
| ASR | Sinhala | CER ↓ | 0.164 ± 0.002 |
| ASR | Tamil | WER ↓ | 0.802 ± 0.006 |
| ASR | Tamil | CER ↓ | 0.296 ± 0.002 |
| Emotion recognition | Tamil | Accuracy ↑ | 0.405 ± 0.018 |
| Emotion recognition | Tamil | Macro-F1 ↑ | 0.387 ± 0.033 |
| Speaker identification | Tamil | Accuracy ↑ | 0.992 ± 0.003 |
| Speaker identification | Tamil | Macro-F1 ↑ | 0.992 ± 0.002 |
| Speaker verification | Tamil | EER ↓ | 0.154 ± 0.019 |
| Speaker diarization | Sinhala | DER ↓ | 0.111 ± 0.011 |
| Speaker diarization | Tamil | DER ↓ | 0.085 ± 0.026 |
| Intent · banking | Sinhala | Accuracy ↑ | 0.987 ± 0.002 \* |
| Intent · banking | Sinhala | Macro-F1 ↑ | 0.987 ± 0.002 \* |
| Intent · banking | Tamil | Accuracy ↑ | 0.767 ± 0.007 |
| Intent · banking | Tamil | Macro-F1 ↑ | 0.749 ± 0.008 |
| Intent · health | Tamil | Accuracy ↑ | 0.487 ± 0.009 |
| Intent · health | Tamil | Macro-F1 ↑ | 0.476 ± 0.008 |

\* Optimistic: no speaker information, so the same speakers can be in train and test.

Per-seed values are in [`results/reference_xlsr300m_norm_v0.2.csv`](results/reference_xlsr300m_norm_v0.2.csv).

### Reading these numbers

- **Diarization DER is almost entirely speaker confusion.** With oracle speech regions,
  missed speech is about 1% (overlap) and false alarm is 0.
- **Speaker identification is near ceiling** over 40 Tamil speakers. It mainly separates
  weak upstreams from adequate ones.
- **Run-to-run variation.** ASR's BiLSTM uses non-deterministic GPU kernels, so re-running
  the same seed moves WER by about ±0.01. Differences smaller than that, or than the
  seed s.d., should not be read as real.
- **Scale differs by language.** Absolute WER, CER and accuracy depend on the corpus and
  script, so compare upstreams within a task, not tasks with each other.

## Not comparable: protocol v0.1

v0.1 trained a single linear layer for a fixed 10 epochs with no dev set. Its splits also
leaked: Sinhala ASR test speakers were also in training, and verification was trained on
its own test speakers. v0.1 scores, including the frozen-encoder baselines produced with
it, must not be put next to v0.2 scores. See the [changelog](../CHANGELOG.md).

## Regenerating the figures and tables

Every number on this page and in the README is read from the versioned CSVs in
[`results/`](results/):

```bash
python docs/scripts/make_figures.py --print        # figures + tables from the CSVs
python docs/scripts/make_figures.py --refresh results/<run> [results/<rerun>] --print
                                                   # rebuild the CSVs from data/ and run outputs
```

`--refresh` takes one or more run directories; a later one replaces an earlier one task by
task. This run's diarization was re-run after a data fix, so it combines
`results/xlsr300m_norm_v0.2_final` and `results/xlsr300m_norm_v0.2_final_sd`.
