import os
from datetime import datetime

import torch
import torchaudio
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
import faster_whisper
from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

from helper_funcs import langs_to_iso, create_config,\
    get_words_speaker_mapping, get_realigned_ws_mapping_with_punctuation, get_sentences_speaker_mapping, \
    get_speaker_aware_transcript, write_srt, cleanup
from services.model_manager.whisper_manager import preload_models
from settings import enable_stemming, audio_path, device, batch_size
from utils.stemming import extract_vocals

#-----------------------------------------------------------------------------------------------------------------------

vocal_target = extract_vocals(audio_path, enable_stemming=enable_stemming, device=device)
audio_waveform = faster_whisper.decode_audio(vocal_target)

#-----------------------------------------------------------------------------------------------------------------------

whisper_model_manager = preload_models()
full_transcript, info = whisper_model_manager.transcribe_file(audio_waveform=audio_waveform)

#-----------------------------------------------------------------------------------------------------------------------

alignment_model, alignment_tokenizer = load_alignment_model(
    device,
    dtype=torch.float16 if device == "cuda" else torch.float32,
)

audio_waveform = (
    torch.from_numpy(audio_waveform)
    .to(alignment_model.dtype)
    .to(alignment_model.device)
)

emissions, stride = generate_emissions(
    alignment_model, audio_waveform, batch_size=batch_size
)

# del alignment_model
# torch.cuda.empty_cache()

tokens_starred, text_starred = preprocess_text(
    full_transcript,
    romanize=True,
    language=langs_to_iso[info.language],
)

segments, scores, blank_token = get_alignments(
    emissions,
    tokens_starred,
    alignment_tokenizer,
)
print("--------------------------------------------------------------------------------")
print("SEGMENTS", segments)
print("SCORES", scores)
print("BLANK_TOKEN", blank_token)
print("--------------------------------------------------------------------------------")


spans = get_spans(tokens_starred, segments, blank_token)
print("SPANS", spans)
word_timestamps = postprocess_results(text_starred, spans, stride, scores)

ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
torchaudio.save(
    os.path.join(temp_path, "mono_file.wav"),
    audio_waveform.cpu().unsqueeze(0).float(),
    16000,
    channels_first=True,
)

#-----------------------------------------------------------------------------------------------------------------------

# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path))
msdd_model.diarize()

# del msdd_model
# torch.cuda.empty_cache()

#-----------------------------------------------------------------------------------------------------------------------
# Reading timestamps <> Speaker Labels mapping

speaker_ts = []
with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(" ")
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

print("SPEAKER_TS", speaker_ts)

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

wsm = get_realigned_ws_mapping_with_punctuation(wsm)
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)


def generate_readable_filename(original_filename: str) -> str:
    base_name, ext = os.path.splitext(f"results/segmentations_and_transcriptions/{original_filename}")
    timestamp = datetime.now().strftime("%Y.%m.%d---%H:%M:%S")
    return f"{base_name}_{timestamp}{ext}"

filename = generate_readable_filename("audio_path.csv")

with open(f"{os.path.splitext(filename)[0]}.txt", "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f)

with open(f"{os.path.splitext(filename)[0]}.srt", "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt)

cleanup(temp_path)