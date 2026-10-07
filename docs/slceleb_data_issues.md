# SLCeleb: duplicated audio in the Sinhala dev and test sets

**Dataset:** SLCeleb for Speaker Verification (IEEE DataPort, DOI [10.21227/smmf-e298](https://doi.org/10.21227/smmf-e298))
**Copy examined:** the full database from the dataset's Google Drive folder — 65,103 files, 19.27 GB
**Date:** October 2026

## Summary

In the copy we downloaded, the **Sinhala** part cannot be used for speaker
verification:

- `sinhala/test` holds 4,486 wav files under 39 speaker IDs, but only
  **1,064 distinct recordings**. Every recording is filed under **3 to 8
  different speaker IDs**.
- All **5,872** clips in `sinhala/dev` are identical to clips in
  `sinhala/test`, filed under different IDs — so the Sinhala dev and test sets
  are the same audio.
- In `test_list_sinhala.txt`, 17 trials compare a recording with itself
  (13 of them labelled as different speakers), and 702 of the 18,860
  different-speaker trials pair two IDs that share recordings.

The **Tamil** part looks sound: no recording appears under more than one
speaker ID, and dev and test share no audio.

"Identical" below means the decoded 16-bit audio samples are equal (MD5 of
the samples), so differences in file headers cannot hide or create a match.

## 1. Sinhala test: each recording appears under 3–8 speaker IDs

| | Count |
|---|---|
| Speaker IDs | 39 (the DataPort page lists 40) |
| wav files | 4,486 (the DataPort page lists 4,620 utterances) |
| Distinct recordings | **1,064** |
| Recordings held by 3 / 4 / 5 / 6 / 7 / 8 IDs | 397 / 414 / 98 / 24 / 69 / 62 |

The duplication follows whole groups of IDs (17 groups in total). The largest:

| Recordings | Speaker IDs holding identical copies |
|---|---|
| 360 | id00005, id00023, id00034 |
| 136 | id00007, id00019, id00025, id00038 |
| 111 | id00002, id00022, id00028, id00035 |
| 84 | id00008, id00011, id00027, id00040 |
| 65 | id00003, id00012, id00026, id00029, id00036 |
| 62 | id00004, id00007, id00017, id00019, id00024, id00025, id00037, id00038 |
| 53 | id00001, id00018, id00021, id00031 |
| 40 | id00009, id00010, id00014, id00020, id00032, id00033, id00039 |

Example: `interview/interview-01-049.wav` is identical in id00009, id00010,
id00014, id00020, id00032, id00033 and id00039. The segment timestamp files
are identical too — e.g. `id00009/interview/interview-01.txt` and
`id00032/interview/interview-01.txt` list the same 50 segments — so whole
speaker folders appear to have been copied under new IDs.

Some IDs belong to more than one group (e.g. id00007, id00008), so the
duplication cannot be undone by merging IDs.

The nested `sinhala/Sinhala_test.zip` has the same content and the same
duplication.

## 2. Sinhala dev: every clip is a copy of a test clip

| | Count |
|---|---|
| Speaker IDs | 51 (the DataPort page lists 110) |
| wav files | 5,872 (the DataPort page lists 12,650 utterances) |
| Distinct recordings | 1,064 — the same 1,064 as `sinhala/test` |
| Clips identical to a `sinhala/test` clip | **5,872 of 5,872** |

Example: `sinhala/dev/id00076/speech/speech-02-003.wav` is identical to
`sinhala/test/id00003/speech/speech-02-003.wav`. Dev ID id00076 holds the same
recordings as test IDs id00003, id00012, id00026, id00029 and id00036. One ID,
id00040, appears in both dev and test.

A model trained on Sinhala dev has therefore heard every Sinhala test
recording.

## 3. Effect on `test_list_sinhala.txt`

| | Count |
|---|---|
| Trials | 37,720 (18,860 same-speaker, 18,860 different-speaker) |
| Trials whose two clips are identical audio | 17 — 13 labelled different-speaker (0) |
| Different-speaker trials whose two IDs share recordings | 702 of 18,860 |

Example: `0 id00021/interview/interview-02-013.wav id00031/interview/interview-02-013.wav`
compares two identical files but is labelled as different speakers. More
generally, with 39 IDs reduced to copies of a few underlying folders, the
speaker labels in the Sinhala trial list cannot be trusted, and EER measured
on it does not reflect speaker verification.

## 4. Tamil: sound, with one count question

| | Speaker IDs | wav files | Distinct recordings | Recordings under >1 ID |
|---|---|---|---|---|
| `Tamil/dev` | 89 (DataPort: 100) | 35,201 (DataPort: 12,100 utterances) | 35,008 | 0 |
| Tamil test (IDs id100xx) | 40 | 4,999 (DataPort: 4,730 utterances) | — | 0 |

No Tamil dev recording appears in the test set, and no trial in
`test_list_tamil.txt` compares identical audio. The only questions are the
counts: `Tamil/dev` has about three times the utterances listed on DataPort,
and 193 of its clips repeat another clip of the same speaker.

Layout notes: `Tamil/test` contains both the 40 Tamil test IDs (id100xx) and
the 39 Sinhala test IDs (id000xx), and a further folder `Tamil -Test` is an
exact copy of `Tamil/test`.

## Questions

1. Is there a corrected version of the Sinhala dev and test sets — with the
   110 dev and 40 test speakers listed on DataPort?
2. Do the DataPort counts describe a different release from the one in the
   Google Drive folder?
3. Are the source video IDs available? The timestamp files would then allow the
   segments to be re-extracted.

## Reproducing the check

```python
import hashlib, collections
from pathlib import Path
import soundfile as sf

def audio_md5(path):
    samples, _ = sf.read(str(path), dtype="int16")
    return hashlib.md5(samples.tobytes()).hexdigest()

test = Path("SLCeleb/sinhala/test")
ids_holding = collections.defaultdict(set)
for wav in test.rglob("*.wav"):
    ids_holding[audio_md5(wav)].add(wav.relative_to(test).parts[0])

print("wav files:", sum(1 for _ in test.rglob("*.wav")))
print("distinct recordings:", len(ids_holding))
print("recordings under >1 speaker ID:", sum(len(v) > 1 for v in ids_holding.values()))
```
