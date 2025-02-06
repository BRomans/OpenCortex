import joblib
import numpy as np

from opencortex.neuroengine.base_model import BaseModelInterface


class SklearnModel(BaseModelInterface):

    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, x):
        return self.model.predict(x)

    def train(self, x: np.ndarray, y: np.ndarray):
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        pass

    def get_params(self) -> dict:
        pass

    def set_params(self, params: dict):
        pass

    def get_name(self) -> str:
        pass

    def get_description(self) -> str:
        pass

