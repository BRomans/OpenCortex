import torch
import numpy as np
from opencortex.neuroengine.base_model import BaseModelInterface


class PyTorchModelInterface(BaseModelInterface):

    def __init__(self, model_path: str):
        self.model = torch.jit.load(model_path)
        self.model.eval()

    def predict(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor_x = torch.tensor(x, dtype=torch.float32)
            return self.model(tensor_x).numpy()

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


