import numpy as np


def convert_to_serializable(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
