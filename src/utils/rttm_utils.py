import os

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class RttmUtils:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.rttm_path = None


    def check_the_results_of_diarization(self):
        rttm_path = os.path.join(self.out_dir, "pred_rttms", "recording_16000hz.rttm")
        if not os.path.isfile(rttm_path):
            logger.error(f"RTTM not found at {rttm_path}. Diarization failed or config error.")
            raise

        self.rttm_path = rttm_path
        return rttm_path


    def format_diarization_rttm(self):
        diar_segments = []
        with open(self.rttm_path, "r") as f:
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

        return diar_segments


    def merging_diarization_with_wshiper_segments(self, diar_segments, whisper_segments):
        final_merged = []
        i, j = 0, 0
        while i < len(whisper_segments) and j < len(diar_segments):
            w = whisper_segments[i]
            d = diar_segments[j]

            overlap_start = max(w["start"], d["start"])
            overlap_end = min(w["end"], d["end"])

            if overlap_end > overlap_start:
                # Створюємо "перетин", додаємо окремим куском
                final_merged.append({
                    "start": overlap_start,
                    "end": overlap_end,
                    "speaker": d["speaker"],
                    "text": w["text"]
                })

            # рухаємося вперед у тому масиві, де "end" менше
            if w["end"] < d["end"]:
                i += 1
            else:
                j += 1
        return final_merged


    def create_result(self, final_merged):
        with open(
                "/Users/antonandreev/python_prog/test_nemo_cpu_mac/results/segmentations_and_transcriptions/result_output.txt",
                "w", encoding="utf-8") as f:
            for seg in final_merged:
                st = seg["start"]
                en = seg["end"]
                sp = seg["speaker"]
                tx = seg["text"]
                print(f"{sp} [{st:.2f}-{en:.2f}]: {tx}")
                f.write(f"{sp} [{st:.2f}-{en:.2f}]: {tx}\n")

