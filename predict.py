import numpy as np
from build_model import BuildModel


class OstroPredict:
    def __init__(self, model_names, batch_size):
        self.model_names = model_names
        self.batch_size = batch_size
        self.models = []
        for model_name in model_names:
            self.models.append(BuildModel(model_name))
        self.num_models = len(self.models)

    def predict(self, img):
        ensemble_prediction = np.zeros((img.shape[0], img.shape[1], 2))
        img = img.reshape([1, img.shape[0], img.shape[1], 3]).astype(float)
        for model in self.models:
            prediction = model.predict((img / 255).astype(float))
            ensemble_prediction += prediction[0] / self.num_models

        ensemble_prediction = (255 * np.argmin(ensemble_prediction, axis=-1)).astype(np.uint8)
        ensemble_prediction = np.squeeze(ensemble_prediction)
        return ensemble_prediction