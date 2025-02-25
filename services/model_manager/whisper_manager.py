import logging
import threading
from functools import lru_cache

import faster_whisper
from faster_whisper import WhisperModel

from helper_funcs import find_numeral_symbol_tokens
from settings import suppress_numerals, whisper_model_name, device, compute_type, batch_size, beams_size


class WhisperModelManager:
    def __init__(self):
        self.model = whisper_model_name
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.beams_size = beams_size

        self.loaded_models = {}
        self._lock = threading.Lock()

    def load_model(self, model_name: str) -> WhisperModel:
        with self._lock:
            if model_name not in self.loaded_models:
                self.loaded_models[model_name] = WhisperModel(
                    model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    num_workers=10,
                )
            return self.loaded_models[model_name]


    def get_model(self, model_name: str) -> WhisperModel:
        return self.load_model(model_name)


    def transcribe_file(self, audio_waveform, model_name: str = whisper_model_name):
        whisper_model = self.get_model(model_name)

        whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)

        suppress_tokens = (
            find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
            if suppress_numerals
            else [-1]
        )

        try:
            transcript_segments, info = whisper_pipeline.transcribe(
                audio_waveform,
                suppress_tokens=suppress_tokens,
                batch_size=self.batch_size,
                without_timestamps=True,
            )
        except Exception as e:
            raise e

        full_transcript = "".join(segment.text for segment in transcript_segments)

        print(full_transcript)

        return full_transcript, info


@lru_cache
def get_model_manager() -> WhisperModelManager:
    return WhisperModelManager()


def preload_models():
    manager = get_model_manager()
    model_name = whisper_model_name

    print(f"Loading model: {model_name}")
    try:
        manager.load_model(model_name)
        print(f"Model {model_name} loaded successfully.")
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        print(e)
        raise

    return manager
