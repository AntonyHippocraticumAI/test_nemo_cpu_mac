import torch

# Name of the audio file
audio_path = "/Users/antonandreev/python_prog/test_nemo_cpu_mac/audio_samples/recording_16000hz.wav"

# Whether to enable music removal from speech, helps increase diarization quality but uses alot of ram
enable_stemming = True

# (choose from 'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large')
whisper_model_name = "large-v3"

# replaces numerical digits with their pronounciation, increases diarization accuracy
suppress_numerals = True

batch_size = 8

language = None  # autodetect language

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)