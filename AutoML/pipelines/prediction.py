import os
import pickle
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.model_path = self.get_model_path()
        self.model = pickle.load(open(self.model_path, 'rb'))

    def get_model_path(self):
        models = os.listdir('artifacts/model_trainer')
        model_path = Path('artifacts/model_trainer') / models[0]
        return model_path
        
    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction