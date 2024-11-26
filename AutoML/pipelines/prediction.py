import os
import pickle
from pathlib import Path

import joblib

class PredictionPipeline:
    def __init__(self):
        self.model_path = self.get_model_path()
        self.model = joblib.load(self.model_path)

    def get_model_path(self):
        models = os.listdir('artifacts/model_trainer')
        model_path = Path('artifacts/model_trainer') / models[0]
        return model_path
        
    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction