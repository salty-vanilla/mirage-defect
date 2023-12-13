import numpy as np
from PIL import Image
from .base import MaskGenerator


class UniformMaskGenerator(MaskGenerator):
    def predict(self, x: Image.Image, *args, **kwargs) -> np.ndarray:
        return np.ones((1, 1), dtype=np.uint8)
