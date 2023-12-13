from PIL import Image
import numpy as np
from mirage_defect.grounded_sam import GroundedSAM
from .base import MaskGenerator


class GroundendSamMaskGenerator(MaskGenerator):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        self.model = GroundedSAM(device)

    def predict(self, x: Image.Image, caption: str) -> np.ndarray:
        mask: np.ndarray = self.model.predict(x, caption)  # type: ignore
        return np.sum(mask, axis=0).astype(dtype=np.bool_).astype(dtype=np.uint8)
