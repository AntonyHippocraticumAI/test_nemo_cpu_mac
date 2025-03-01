import os
import json
import librosa
import torch
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf

# FASTER-WHISPER
from faster_whisper import WhisperModel, BatchedInferencePipeline

# NeMo Diarization
from nemo.collections.asr.models import NeuralDiarizer

##############################################################################
# 1) НАЛАШТУВАННЯ
##############################################################################
# Ваше аудіо
AUDIO_PATH = "/Users/antonandreev/python_prog/test_nemo_cpu_mac/audio_samples/recording_16000hz.wav"

# Whisper
WHISPER_MODEL_NAME = "large-v3"

# NeMo Speaker Diarization параметри (VAD, Embeddings, MSDD)
VAD_MODEL_PATH = "/Users/antonandreev/python_prog/test_nemo_cpu_mac/nemo_vad_models/vad_marblenet.nemo"   # або локальний .nemo файл
EMBED_MODEL_PATH = "/Users/antonandreev/titanet_large_vv1/titanet-l.nemo"             # або "/path/to/titanet-l.nemo"
# MSDD може автоматично взяти дефолтну модель, якщо enable_msdd=True.
# Або можна вказати свою модель у config.

# Кількість спікерів
# Якщо знаєте точно 2, можна вимкнути auto і поставити kmeans_num_clusters=2
FORCE_SPEAKERS = None  # або 2

# Де зберігатимуться проміжні результати NeMo Diarizer
ROOT_DIR = "./nemo_diar_temp"
os.makedirs(ROOT_DIR, exist_ok=True)

MANIFEST_PATH = os.path.join(ROOT_DIR, "input_manifest.json")
OUT_DIR = os.path.join(ROOT_DIR, "diar_output")


def main():
    ##########################################################################
    # A) ТРАНСКРИПЦІЯ WHISPER
    ##########################################################################
    audio, sr = librosa.load(AUDIO_PATH, sr=16000)
    print(f"Loaded audio {AUDIO_PATH}, sr={sr}, duration={len(audio)/sr:.1f}s")

    print("=== WHISPER STEP ===")
    whisper_model = WhisperModel(
        WHISPER_MODEL_NAME,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float16" if torch.cuda.is_available() else "int8",
    )
    pipeline = BatchedInferencePipeline(whisper_model)

    segments_gen, _ = pipeline.transcribe(audio, beam_size=1, batch_size=8, without_timestamps=False)

    whisper_segments = []
    for seg in segments_gen:
        whisper_segments.append({
            "start": seg.start,
            "end":   seg.end,
            "text":  seg.text.strip().replace("\n", " "),
        })

    print(f"Whisper produced {len(whisper_segments)} segments.")

    ##########################################################################
    # B) NEMO DIARIZATION (MarbleNet -> Segmentation -> TitaNet -> Clustering -> MSDD)
    ##########################################################################

    # 1) Створюємо маніфест (JSON) на одне аудіо:
    manifest_entry = {
        "audio_filepath": AUDIO_PATH,
        "offset": 0,
        "duration": None,
        "label": "infer",  # умовно
        "text": "-",
        "num_speakers": FORCE_SPEAKERS,  # Якщо точно 2, пишемо 2
        "rttm_filepath": None,
        "uem_filepath": None
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest_entry) + "\n")

    # 2) Створюємо config для NeuralDiarizer
    #    enable_msdd=True -> Увімкнути MSDD (Multi-scale diarization decoder)
    #    Увага: Якщо виникнуть проблеми, можливо, доведеться вказати msdd_model вручну.
    diar_config = {
        "device": None,
        "num_workers": 0,
        "batch_size": 64,
        "sample_rate": 16000,
        "verbose": True,

        "diarizer": {
            "manifest_filepath": MANIFEST_PATH,
            "out_dir": OUT_DIR,
            "oracle_vad": False,  # використ. системний VAD
            "collar": 0.25,
            "ignore_overlap": True,

            "vad": {
                "model_path": VAD_MODEL_PATH,
                'external_vad_manifest': None,
                "parameters": {
                    "window_length_in_sec": 0.15,
                    "smoothing": "median",  # False or type of smoothing method (eg: median)
                    "overlap": 0.5,  # Overlap ratio for overlapped mean/median smoothing filter
                    "onset": 0.8,  # Onset threshold for detecting the beginning and end of a speech
                    "offset": 0.6,  # Offset threshold for detecting the end of a speech
                    "pad_onset": 0.1,  # Adding durations before each speech segment
                    "pad_offset": -0.05,  # Adding durations after each speech segment
                    "min_duration_on": 0,  # Threshold for small non_speech deletion
                    "min_duration_off": 0.2,  # Threshold for short speech segment deletion
                    "filter_speech_first": True,
                    "shift_length_in_sec": 0.01,
                }
            },
            "speaker_embeddings": {
                "model_path": EMBED_MODEL_PATH,
                "infer_batches": True,
                "parameters": {
                    "window_length_in_sec": [1.5,1.25,1.0,0.75,0.5], # Window length(s) in sec (floating-point number). either a number or a list. ex) 1.5 or [1.5,1.0,0.5]
                    "shift_length_in_sec": [0.75,0.625,0.5,0.375,0.25], # Shift length(s) in sec (floating-point number). either a number or a list. ex) 0.75 or [0.75,0.5,0.25]
                    "multiscale_weights": [1,1,1,1,1], # Weight for each scale. should be null (for single scale) or a list matched with window/shift scale count. ex) [0.33,0.33,0.33]
                    "save_embeddings": True # If True, save speaker embeddings in pickle format. This should be True if clustering result is used for other models, such as `msdd_model`.


                }
            },
            "clustering": {
                # Якщо точно 2 спікери:
                # "kmeans_num_clusters": 2,
                # "oracle_num_speakers": False,

                # Якщо auto:
                "kmeans_num_clusters": None,
                "oracle_num_speakers": False,
                "parameters": {
                    "oracle_num_speakers": False,  # If True, use num of speakers value provided in manifest file.
                    "max_num_speakers": 8,
                    # Max number of speakers for each recording. If an oracle number of speakers is passed, this value is ignored.
                    "enhanced_count_thres": 80,
                    # If the number of segments is lower than this number, enhanced speaker counting is activated.
                    "max_rp_threshold": 0.25,  # Determines the range of p-value search: 0 < p <= max_rp_threshold.
                    "sparse_search_volume": 30,  # The higher the number, the more values will be examined with more time.
                    "maj_vote_spk_count": False,
                    # If True, take a majority vote on multiple p-values to estimate the number of speakers.
                    "chunk_cluster_count": 50,
                    # Number of forced clusters (overclustering) per unit chunk in long-form audio clustering.
                    "embeddings_per_chunk": 10000
                    # Number of embeddings in each chunk for long-form audio clustering. Adjust based on GPU memory capacity. (default: 10000, approximately 40 mins of audio)
                }
            },
            "msdd_model": {
                # Параметри MSDD. Якщо треба можна вказати "model_path": "diar_msdd_telephonic..."
                # або лишити порожнім, тоді NeMo завантажить дефолтну модель.
                "model_path": "diar_msdd_telephonic",
                "parameters": {
                    "use_speaker_model_from_ckpt": True,
                    # If True, use speaker embedding model in checkpoint. If False, the provided speaker embedding model in config will be used.
                    "infer_batch_size": 25,  # Batch size for MSDD inference.
                    "sigmoid_threshold": [0.7],
                    # Sigmoid threshold for generating binarized speaker labels. The smaller the more generous on detecting overlaps.
                    "seq_eval_mode": False,
                    # If True, use oracle number of speaker and evaluate F1 score for the given speaker sequences. Default is False.
                    "split_infer": True,
                    # If True, break the input audio clip to short sequences and calculate cluster average embeddings for inference.
                    "diar_window_length": 50,  # The length of split short sequence when split_infer is True.
                    "overlap_infer_spk_limit": 5,
                    # If the estimated number of speakers are larger than this number, overlap speech is not estimated.

                }
            },
            "enable_msdd": True,  # вмикаємо MSDD
            "convert_to_unique_speaker_ids": True,  # видаватиме speaker_0, speaker_1, ...
        }
    }

    cfg = OmegaConf.create(diar_config)
    # 3) Запускаємо NeMo Diarizer
    print("=== NeMo DIARIZATION (with MSDD) ===")
    diar_model = NeuralDiarizer(cfg=cfg)
    diar_model.diarize()

    # Результати діаризації в RTTM:
    rttm_path = os.path.join(OUT_DIR, "pred_rttms", "recording_16000hz.rttm")
    if not os.path.isfile(rttm_path):
        print(f"RTTM not found at {rttm_path}. Diarization failed or config error.")
        return

    ##########################################################################
    # C) ЗЧИТАТИ RTTM і злити з Whisper-сегментами
    ##########################################################################
    # Формат: SPEAKER <audio_name> 1 <start_time> <duration> <..> <..> <speaker_label>
    diar_segments = []
    with open(rttm_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        # Пропустимо неправильні строки
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start_time = float(parts[3])
        duration = float(parts[4])
        spk_label = parts[7]
        diar_segments.append({
            "start": start_time,
            "end":   start_time + duration,
            "speaker": spk_label
        })

    # Сортуємо за часом
    diar_segments.sort(key=lambda x: x["start"])

    # Тепер злити diar_segments із whisper_segments:
    # Кожен Whisper-сегмент (wseg) може перекриватися з кількома diar-сегментами.
    final_merged = []
    i, j = 0, 0
    while i < len(whisper_segments) and j < len(diar_segments):
        w = whisper_segments[i]
        d = diar_segments[j]

        overlap_start = max(w["start"], d["start"])
        overlap_end   = min(w["end"],   d["end"])

        if overlap_end > overlap_start:
            # Створюємо "перетин", додаємо окремим куском
            final_merged.append({
                "start":   overlap_start,
                "end":     overlap_end,
                "speaker": d["speaker"],
                "text":    w["text"]
            })

        # рухаємося вперед у тому масиві, де "end" менше
        if w["end"] < d["end"]:
            i += 1
        else:
            j += 1

    ##########################################################################
    # D) Виводимо фінальний мердж
    ##########################################################################
    print("\n=== FINAL MERGED (WHISPER + MSDD DIARIZATION) ===")
    with open("/Users/antonandreev/python_prog/test_nemo_cpu_mac/src/nemo_diar_temp/diar_output/pred_rttms/result_output.txt", "w", encoding="utf-8") as f:
        for seg in final_merged:
            st = seg["start"]
            en = seg["end"]
            sp = seg["speaker"]
            tx = seg["text"]
            print(f"{sp} [{st:.2f}-{en:.2f}]: {tx}")
            f.write(f"{sp} [{st:.2f}-{en:.2f}]: {tx}\n")


if __name__ == "__main__":
    main()
