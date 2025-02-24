import os
from datetime import datetime

import torch
import torchaudio
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
import logging
import faster_whisper
from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

from helper_funcs import find_numeral_symbol_tokens, langs_to_iso, create_config, punct_model_langs, \
    get_words_speaker_mapping, get_realigned_ws_mapping_with_punctuation, get_sentences_speaker_mapping, \
    get_speaker_aware_transcript, write_srt, cleanup
from settings import enable_stemming, audio_path, device, whisper_model_name, suppress_numerals, batch_size, language

#-----------------------------------------------------------------------------------------------------------------------

if enable_stemming:
    # Isolate vocals from the rest of the audio

    return_code = os.system(
        f'python -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o "temp_outputs" --device "{device}"'
    )

    if return_code != 0:
        logging.warning("Source splitting failed, using original audio file.")
        vocal_target = audio_path
    else:
        vocal_target = os.path.join(
            "temp_outputs",
            "htdemucs",
            os.path.splitext(os.path.basename(audio_path))[0],
            "vocals.wav",
        )
else:
    vocal_target = audio_path

#-----------------------------------------------------------------------------------------------------------------------

compute_type = "int8"
# or run on GPU with INT8
# compute_type = "int8_float16"
# or run on CPU with INT8
# compute_type = "int8"

whisper_model = faster_whisper.WhisperModel(
    whisper_model_name, device=device, compute_type=compute_type
)
whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
audio_waveform = faster_whisper.decode_audio(vocal_target)
suppress_tokens = (
    find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
    if suppress_numerals
    else [-1]
)

if batch_size > 0:
    transcript_segments, info = whisper_pipeline.transcribe(
        audio_waveform,
        language,
        suppress_tokens=suppress_tokens,
        batch_size=batch_size,
        without_timestamps=True,
    )
else:
    transcript_segments, info = whisper_model.transcribe(
        audio_waveform,
        language,
        suppress_tokens=suppress_tokens,
        without_timestamps=True,
        vad_filter=True,
    )

full_transcript = "".join(segment.text for segment in transcript_segments)
print(full_transcript)
# clear gpu vram
del whisper_model, whisper_pipeline
torch.cuda.empty_cache()

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

del alignment_model
torch.cuda.empty_cache()

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

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")


if info.language in punct_model_langs:
    # restoring punctuation in the transcript to help realign the sentences
    punct_model = PunctuationModel(model="kredor/punctuate-all")

    words_list = list(map(lambda x: x["word"], wsm))

    labled_words = punct_model.predict(words_list, chunk_size=230)

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"

    # We don't want to punctuate U.S.A. with a period. Right?
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

    for word_dict, labeled_tuple in zip(wsm, labled_words):
        word = word_dict["word"]
        if (
            word
            and labeled_tuple[1] in ending_puncts
            and (word[-1] not in model_puncts or is_acronym(word))
        ):
            word += labeled_tuple[1]
            if word.endswith(".."):
                word = word.rstrip(".")
            word_dict["word"] = word

else:
    logging.warning(
        f"Punctuation restoration is not available for {info.language} language. Using the original punctuation."
    )

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