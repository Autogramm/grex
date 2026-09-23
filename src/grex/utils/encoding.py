import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    An encoder that transforms numpy objects into a valid json-compatible format.
    """
    def default(self, obj):
        # numpy floats, integers, and booleans to their native Python types
        if isinstance(obj, (np.generic, np.ndarray)):
            return obj.item() if obj.ndim == 0 else obj.tolist()
        return json.JSONEncoder.default(self, obj)
