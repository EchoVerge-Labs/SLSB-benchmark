"""Task family registry. A "family" is the CLI-facing short name (--tasks
asr,sid,er,sd,asv); it maps to the data/<prefix>* directory naming convention
and to the module that knows how to run it.
"""
from slsb.tasks import asr, emotion, sid, speaker_diarization, speaker_verification

FAMILY_DIR_PREFIXES = {
    "asr": ("asr",),
    "sid": ("sid",),
    "er": ("er_",),
    "sd": ("sd_",),
    "asv": ("asv",),
}

# Data folders that match a family's prefix but are not benchmarked. Their data
# stays in data/ (DVC-tracked, and v0.1 results used it); they are just never run.
EXCLUDED_TASK_DIRS = {
    # One speaker only, so it can't be split speaker-disjointly: a speaker-dependent
    # test that isn't comparable with asr_sinhala / asr_tamil. Excluded in v0.2.
    "asr_omni_sinhala",
}

FAMILY_MODULES = {
    "asr": asr,
    "sid": sid,
    "er": emotion,
    "sd": speaker_diarization,
    "asv": speaker_verification,
}

__all__ = ["EXCLUDED_TASK_DIRS", "FAMILY_DIR_PREFIXES", "FAMILY_MODULES", "asr", "emotion", "sid",
           "speaker_diarization", "speaker_verification"]
