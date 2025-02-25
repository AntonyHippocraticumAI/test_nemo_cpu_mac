import nemo.collections.asr as nemo_asr

# Вказуємо шлях до файлу моделі
vad_model_path = "/Users/antonandreev/vad_telephony_marblenet_v1.0.0rc1/vad_telephony_marblenet.nemo"

# Завантажуємо модель локально
vad_model = nemo_asr.models.EncDecClassificationModel.restore_from(vad_model_path)

# Перевіряємо завантаження
print(vad_model)
