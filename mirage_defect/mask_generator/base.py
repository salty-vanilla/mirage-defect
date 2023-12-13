from abc import abstractmethod
import numpy as np
from PIL import Image


class MaskGenerator(object):
    @abstractmethod
    def predict(self, x: Image.Image, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError
