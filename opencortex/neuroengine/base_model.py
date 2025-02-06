from abc import ABC, abstractmethod
import numpy as np


class BaseModelInterface(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """ Runs inference on input data X and returns the predicted output """
        pass

    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray):
        """ Trains the model on input data X and target y """
        pass

    @abstractmethod
    def save(self, path: str):
        """ Saves the model to disk """
        pass

    @abstractmethod
    def load(self, path: str):
        """ Loads the model from disk """
        pass

    @abstractmethod
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        """ Evaluates the model on input data X and target y """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """ Returns a dictionary of hyperparameters """
        pass

    @abstractmethod
    def set_params(self, params: dict):
        """ Sets the hyperparameters of the model """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """ Returns the name of the model """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """ Returns the description of the model """
        pass
