import os
from typing import Union
import urllib.request
from PIL import Image
import numpy as np
import torch
from groundingdino.util.inference import load_model, predict
from groundingdino.config import GroundingDINO_SwinT_OGC
import groundingdino.datasets.transforms as T
from segment_anything import sam_model_registry, SamPredictor
from mirage_defect.logger import logger


GROUNDING_DINO_WEIGHTS_URI = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
)
GROUNDING_DINO_WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
SAM_WEIGHTS_URI = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_WEIGHTS_NAME = "sam_vit_h_4b8939.pth"
SAM_VERSION = "vit_h"


class GroundedSAM(object):
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        groundind_dino_weights_path, sam_weights_path = self._download_models()
        self.grounding_dino = load_model(
            GroundingDINO_SwinT_OGC.__file__, groundind_dino_weights_path, device=self.device
        )
        self.sam = SamPredictor(sam_model_registry[SAM_VERSION](checkpoint=sam_weights_path).to(self.device))

    def predict(
        self,
        image: Image.Image,
        caption: str,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        is_return_full: bool = False,
    ) -> Union[tuple[np.ndarray, np.ndarray, list[str], np.ndarray], np.ndarray]:
        image_np, image_tensor = self._preprocess(image)
        image_tensor.to(self.device)
        boxes, logits, phrases = self._get_groundind_dino_output(image_tensor, caption, box_threshold, text_threshold)

        boxes.cpu()
        height, width = image_np.shape[:2]
        for i in range(len(boxes)):
            boxes[i] = boxes[i] * torch.Tensor([width, height, width, height])
            boxes[i][:2] -= boxes[i][2:] / 2
            boxes[i][2:] += boxes[i][:2]

        masks = self._get_sam_output(image_np, boxes)

        if is_return_full:
            return boxes.cpu().numpy(), logits.cpu().numpy(), phrases, masks.cpu().numpy().transpose(1, 0, 2, 3)[0]
        else:
            return masks.cpu().numpy().transpose(1, 0, 2, 3)[0]

    def _get_groundind_dino_output(self, image, caption, box_threshold, text_threshold):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        boxes, logits, phrases = predict(
            model=self.grounding_dino,
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        return boxes, logits, phrases

    def _get_sam_output(self, image, boxes):
        self.sam.set_image(image)
        transformed_boxes = self.sam.transform.apply_boxes_torch(boxes, image.shape[:2]).to(self.device)
        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,  # type: ignore
        )
        return masks

    def _download_models(self):
        def _get_dir_path():
            env_path = os.getenv("MIRAGE_DEFECT_WEIGTHS_DIR")
            if env_path:
                return env_path
            else:
                home = os.path.expanduser("~")
                weights_dir = os.path.join(home, ".mirage-defect")
                os.makedirs(weights_dir, exist_ok=True)
                return weights_dir

        weights_dir = _get_dir_path()

        paths = []

        for uri, name in zip(
            [GROUNDING_DINO_WEIGHTS_URI, SAM_WEIGHTS_URI], [GROUNDING_DINO_WEIGHTS_NAME, SAM_WEIGHTS_NAME]
        ):
            weights_path = os.path.join(weights_dir, name)
            if not os.path.exists(weights_path):
                logger.info(f"Downloading {name} from {uri} ...")
                data = urllib.request.urlopen(uri).read()
                with open(weights_path, mode="wb") as f:
                    f.write(data)
            paths.append(weights_path)

        return paths

    def _preprocess(self, image: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image = image.convert("RGB")
        image_np = np.asarray(image)
        image_tensor, _ = transform(image, None)
        return image_np, image_tensor  # type: ignore
