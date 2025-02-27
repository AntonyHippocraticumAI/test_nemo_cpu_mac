import nemo.collections.asr as nemo_asr

available_models = nemo_asr.models.EncDecClassificationModel.list_available_models()
print(available_models)
