"""Speaker diarization (SD): NOT YET IMPLEMENTED.

data/sd_sinhala/ and data/sd_tamil/ hold paired <name>.wav / <name>.rttm files
(RTTM = NIST Rich Transcription speaker-turn format), not one of the three
labels.csv / transcripts.csv / trials_*.csv conventions the other tasks use --
so slsb.utils.datasets.discover_tasks() correctly does not pick them up
(they show up in find_unrecognized_dirs() instead).

Evaluating this task needs a DER (diarization error rate) runner: segment
each wav, cluster frozen-upstream embeddings into speaker turns, and score
against the reference RTTM (e.g. with pyannote.metrics.diarization.DiarizationErrorRate).
That runner does not exist yet in this codebase (it didn't exist in
Basemodel_Benchmark either -- see data_prep/prep_sita.py's docstring), so
`slsb run --tasks sd` fails loudly here rather than silently skipping or
fabricating a score. See KNOWN_ISSUES.md.
"""
from pathlib import Path


def discover(data_dir: Path, lang: str) -> list[str]:
    """Returns the RTTM basenames staged under data_dir/sd_<lang>/, if any."""
    task_dir = Path(data_dir) / f"sd_{lang}"
    if not task_dir.exists():
        return []
    return sorted(p.stem for p in task_dir.glob("*.rttm"))


def run(upstream, data_dir: Path, lang: str, params, seed=42):
    raise NotImplementedError(
        "speaker diarization (sd) has no DER runner yet -- data/sd_{sinhala,tamil} "
        "are staged (wav+rttm pairs) but not wired into an evaluator. "
        "See slsb/tasks/speaker_diarization.py and KNOWN_ISSUES.md."
    )
