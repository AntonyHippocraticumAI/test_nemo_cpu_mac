from src.services.model_manager.vad_manager import preload_vad_model
from src.services.model_manager.whisper_manager import preload_models
from src.utils.settings import AUDIO_PATH

#-----------------------------------------------------------------------------------------------------------------------

whisper_model_manager = preload_models()
vad_model_manager = preload_vad_model()
print("Loaded VAD and WHISPER models")

#-----------------------------------------------------------------------------------------------------------------------

audio, sr = vad_model_manager.get_audio_time_series(AUDIO_PATH)
print("audio", audio)

segments = vad_model_manager.get_speech_segments(audio, sr)
print("segments", segments)

cleaned_audio = vad_model_manager.get_ndarray_of_segments(audio, segments, sr)
print("cleaned_audio", cleaned_audio)

full_transcript, info = whisper_model_manager.transcribe_with_vad(cleaned_audio, batch_flag=True)
print(full_transcript)