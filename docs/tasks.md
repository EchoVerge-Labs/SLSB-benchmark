# Tasks

The ten tasks run in protocol v0.2, and the three kept in `data/` but not run. Sizes come
from [`results/tasks.csv`](results/tasks.csv), which
[`scripts/make_figures.py --refresh`](scripts/make_figures.py) rebuilds from `data/`.

**Licences.** Each corpus keeps its own licence, and SLSB grants nothing beyond it. Where a
licence could not be confirmed, the table says so. Read the source's terms before using or
redistributing its part of `data/`.

| Corpus | Used by | Licence / access | Source |
|---|---|---|---|
| OpenSLR-52, *Large Sinhala ASR training data set* | `asr_sinhala` | CC BY-SA 4.0 | [openslr.org/52](https://openslr.org/52/) |
| TaLK-Corpus, Sri Lankan Tamil | `asr_tamil` | not confirmed; see source | [ACL Anthology, SPEAKABLE @ LREC 2026](https://aclanthology.org/2026.speakable-1.21/) |
| EmoTa | `er_tamil` | academic use under CC BY-NC 4.0 (EmoTa licence v1.0); gated on Hugging Face; included here with the authors' permission | [github.com/aaivu/EmoTa](https://github.com/aaivu/EmoTa) · [Hugging Face](https://huggingface.co/datasets/aaivu-labs/EmoTa) |
| SLCeleb | `sid`, `asv_tamil` | CC BY 4.0 | [IEEE DataPort, DOI 10.21227/smmf-e298](https://ieee-dataport.org/documents/slceleb-speaker-verification) |
| SiTa | `sd_sinhala`, `sd_tamil` | not confirmed; see source | [ACL Anthology, CHiPSAL 2025](https://aclanthology.org/2025.chipsal-1.8/) |
| Sinhala and Tamil banking intents (crowdsourced) | `ic_banking_*` | academic / research use only, no commercial use (licence file in the archive) | team archive |
| Tamil health intents | `ic_health_tamil` | not confirmed | team archive |

---

## Speech recognition: `asr_sinhala`, `asr_tamil`

| | `asr_sinhala` | `asr_tamil` |
|---|---|---|
| Data | 3,000 utterances sampled from OpenSLR-52, speaker-balanced; read speech | 1,214 utterances from TaLK |
| Audio | 3.6 h | 1.6 h |
| Speakers | 478 (from `utt_spk_text.tsv`) | 22 (inferred from the filename prefix) |
| Split | speaker-disjoint 2,096 / 303 / 601 clips | speaker-disjoint 752 / 123 / 339 clips (the v0.1 test set) |
| Vocabulary | 118 characters | 57 characters |
| Metric | WER (primary), CER | WER (primary), CER |

The Tamil test set holds only 5 speakers, so its scores vary more between models than
the Sinhala ones.

## Emotion recognition: `er_tamil`

936 clips, 0.7 h: 22 speakers each read 19 sentences in five emotions (anger, happiness,
sadness, fear, neutral). Each sentence is recorded in most emotions, so sentence overlap
doesn't reveal the label; speakers are what leak. Scored by speaker-disjoint 5-fold
cross-validation (4–5 speakers per fold). The metrics are accuracy and macro-F1, averaged
over folds.

## Speaker identification: `sid`

A closed-set task over SLCeleb's 40 Tamil test speakers: 4,993 clips, 18 h, of interviews,
speeches and other YouTube videos. Every speaker is a class, so train / dev / test share
speakers by definition. Since v0.4 the split is by **source video**: each speaker has 4–10
videos, about 20% of them (at least one) are test and one is dev, so no test clip shares a
recording session, microphone or room with training (3,209 / 823 / 961 clips; 156 / 40 / 46
videos). With the v0.3 clip-level split every test clip had sibling clips from the same video
in training, and every model scored 0.99+; on unseen videos frozen XLS-R drops to about 0.87.
Repeated audio is kept once. SLCeleb's 39 Sinhala test
speakers are dropped, because their audio is filed under several speaker IDs at once
([details](slceleb_data_issues.md)). The metrics are accuracy and macro-F1.

## Speaker verification: `asv_tamil`

Tested on SLCeleb's Tamil trial list: 37,720 trials, half same-speaker, over 4,999 clips of
40 speakers. The head trains on 80 SLCeleb **dev** speakers (11,853 clips, at most 150 per
speaker) and is selected on 5,400 dev trials from 9 more. No training speaker appears in the
trials, and no training clip is identical to a trial clip. The metric is EER with cosine
scoring.

`asv_sinhala` is not run: every clip in the Sinhala trial list is filed under several speaker
IDs, so its labels can't be trusted.

## Speaker diarization: `sd_sinhala`, `sd_tamil`

| | `sd_sinhala` | `sd_tamil` |
|---|---|---|
| Recordings | 60 (10.0 h), 5–17 min each | 14 (2.0 h) |
| Speakers per recording | 1–10, mostly 2–5 | 2–6, mostly 3 |
| Overlapped speech | 1.5% of speech | 3.0% of speech |
| Split (recordings) | 24 / 12 / 24 | 6 / 3 / 5 |

In-the-wild YouTube audio with RTTM references. The metric is DER (0.25 s collar,
overlap included), with oracle speech regions; see [protocol §5](protocol.md#5-speaker-diarization).
One Sinhala test recording, `YT_49_BS`, has turns 80 s past the end of its audio; they are
clipped to the audio.

## Intent classification: `ic_banking_sinhala`, `ic_banking_tamil`, `ic_health_tamil`

| | `ic_banking_sinhala` | `ic_banking_tamil` | `ic_health_tamil` |
|---|---|---|---|
| Intents | 6 banking (balance, transfer, bill payment, …) | the same 6 | 16 symptoms (cough, headache, back pain, …) |
| Prompts | 39 fixed sentences | 31 fixed sentences | 160 phrases |
| Clips | 7,588 (6.8 h) | 400 (0.4 h) | 1,453 (2.0 h) |
| Speakers | not recorded | 40 | 100 |
| Evaluation | sentence-disjoint 5-fold | sentence-disjoint 5-fold | sentence-disjoint 5-fold |

Built by [`data_prep/prep_intent.py`](../data_prep/prep_intent.py). Clips without a label in
the source CSVs are skipped (74 and 187), identical audio is kept once, and two
Sinhala clips whose copies carry different intents are dropped.

Each dataset is a small set of fixed prompts, each read many times (Banking Sinhala:
147–225 readings per sentence). Until v0.4 every test sentence was also in training, so
the task partly measured recognising a memorised sentence: `ic_banking_sinhala` scored
0.97–0.99 for every model. Since v0.5 the folds are **sentence-disjoint**: the prompts
of a test fold are never heard in training, so a model has to map unseen wording to its
intent. Speakers may recur across folds (Banking Sinhala records none); a speaker's
identity predicts their intent only weakly in Tamil (30% vs 25% majority for banking,
17% vs 7% for health).

---

## Not run in v0.2

| Task | Data | Why |
|---|---|---|
| `asv_sinhala` | SLCeleb Sinhala trials | 4,486 clips are 1,064 recordings copied under 39 speaker IDs ([details](slceleb_data_issues.md)) |
| `asr_omni_sinhala` | `facebook/omnilingual-asr-corpus`, 111 clips | one speaker, so no speaker-disjoint split is possible |
| `er_sinhala` | `jithara/sinhala-emotional-tts-dataset` | broken source metadata; not copied into `data/` |

`asr_omni_sinhala` and `asv_sinhala` are listed in `slsb.tasks.EXCLUDED_TASKS`; their data
stays in `data/`.
