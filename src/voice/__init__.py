"""Real-time voice pipeline: VAD, STT, and TTS for Tamil math tutoring."""

from .stt import STTResult, TamilSTTPipeline
from .tts import TamilTTSPipeline
from .vad import AudioUtterance, VADConfig, VoiceActivityDetector

__all__ = [
    "AudioUtterance",
    "STTResult",
    "TamilSTTPipeline",
    "TamilTTSPipeline",
    "VADConfig",
    "VoiceActivityDetector",
]
