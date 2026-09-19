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

FAMILY_MODULES = {
    "asr": asr,
    "sid": sid,
    "er": emotion,
    "sd": speaker_diarization,
    "asv": speaker_verification,
}

__all__ = ["FAMILY_DIR_PREFIXES", "FAMILY_MODULES", "asr", "emotion", "sid",
           "speaker_diarization", "speaker_verification"]
