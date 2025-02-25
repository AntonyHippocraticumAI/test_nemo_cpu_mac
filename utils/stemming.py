import logging
import os


def extract_vocals(audio_path: str, enable_stemming:bool =True, device: str ='cpu'):
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

    return vocal_target