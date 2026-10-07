"""Task family registry. A "family" is the CLI-facing short name (--tasks
asr,sid,er,sd,asv); it maps to the data/<prefix>* directory naming convention
and to the module that knows how to run it.
"""
from slsb.tasks import asr, emotion, intent, sid, speaker_diarization, speaker_verification

FAMILY_DIR_PREFIXES = {
    "asr": ("asr",),
    "sid": ("sid",),
    "er": ("er_",),
    "sd": ("sd_",),
    "asv": ("asv",),
    "ic": ("ic_",),
}

# Tasks (data folders or ASV trial lists) that are discovered but never run.
# Their data stays in data/ (DVC-tracked, and v0.1 results used it).
EXCLUDED_TASKS = {
    # One speaker only, so it can't be split speaker-disjointly: a speaker-dependent
    # test that isn't comparable with asr_sinhala / asr_tamil.
    "asr_omni_sinhala",
    # SLCeleb's Sinhala test set is 1,064 recordings copied under 39 speaker ids,
    # so its trial labels are unreliable (docs/slceleb_data_issues.md).
    "asv_sinhala",
}

FAMILY_MODULES = {
    "asr": asr,
    "sid": sid,
    "er": emotion,
    "sd": speaker_diarization,
    "asv": speaker_verification,
    "ic": intent,
}

__all__ = ["EXCLUDED_TASKS", "FAMILY_DIR_PREFIXES", "FAMILY_MODULES", "asr", "emotion", "intent", "sid",
           "speaker_diarization", "speaker_verification"]
