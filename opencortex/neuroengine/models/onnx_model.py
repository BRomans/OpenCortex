import onnxruntime as ort

from opencortex.neuroengine.base_model import BaseModelInterface


class OnnxModelInterface(BaseModelInterface):

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, x):
        return self.session.run(None, {self.input_name: X.astype(np.float32)})[0]

    def train(self, x, y):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def evaluate(self, x, y):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    def set_params(self, params):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError

    def get_description(self):
        raise NotImplementedError
