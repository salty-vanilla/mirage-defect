# This code is a modification of the code found at the following GitHub repository:
# https://github.com/hmsch/natural-synthetic-anomalies/blob/main/self_sup_data/self_sup_tasks.py
from typing import NamedTuple, Optional, TypedDict, Union
import cv2
import numpy as np
from PIL import Image
from mirage_defect.logger import logger
from mirage_defect.grounded_sam import GroundedSAM
from mirage_defect.mask_generator import MaskGenerator


MAX_ATTEMPTS = 200
POSITIVE_MASK_THRESHOLD = 50


class Coords(NamedTuple):
    lx: int
    ly: int
    rx: int
    ry: int


class NsaGeneratorArgs(NamedTuple):
    num_patches: int = 1
    mode: str = "swap"
    min_patch_length_rate: float = 0.05
    max_patch_length_rate: float = 0.20
    object_overlap_threshold: float = 0.15
    and_overlap_threshold: float = 0.15
    shift: bool = True
    label_mode: str = "binary"
    resize: bool = True
    gamma_params: Optional[tuple] = None
    intensity_logistic_params: tuple = (1 / 6, 20)
    resize_bounds: tuple = (0.7, 1.3)


class NsaGenerator(object):
    def __init__(self, args: NsaGeneratorArgs, mask_generator: MaskGenerator) -> None:
        self.args = args
        self.mask_generator = mask_generator

    def set_random_args(self):
        self.args = NsaGeneratorArgs()

    def generate(self, x1: Image.Image, x2: Optional[Image.Image] = None, caption: str = "", n: int = 1):
        output_images = []
        output_masks = []
        if x2 is None:
            x2 = x1.copy()

        x1_object_mask = self.mask_generator.predict(x1, caption)
        x2_object_mask = self.mask_generator.predict(x2, caption)

        x1_np = np.asarray(x1)
        x2_np = np.asarray(x2)

        # # (N_MASKS, H, W) -> (H, W)
        # x1_object_mask = np.sum(x1_object_mask, axis=0).astype(dtype=np.uint8)
        # x2_object_mask = np.sum(x2_object_mask, axis=0).astype(dtype=np.uint8)

        x1_object_mask = cv2.resize(x1_object_mask, (x1_np.shape[1], x1_np.shape[0]))
        x2_np = cv2.resize(x2_np, (x1_np.shape[1], x1_np.shape[0]))
        x2_object_mask = cv2.resize(x2_object_mask, (x1_np.shape[1], x1_np.shape[0]))

        for _ in range(n):
            image = np.asarray(x1.copy())

            if self.args.label_mode == "continuous":
                factor = np.random.uniform(0.05, 0.95)
            else:
                factor = 1

            for i in range(self.args.num_patches):
                if i == 0 or np.random.randint(2) > 0:  # at least one patch
                    image = self._update_image(image, x1_np, x2_np, x1_object_mask, x2_object_mask, factor)

            mask = np.array(image - x1_np).sum(axis=-1).astype(dtype=bool).astype(dtype=np.uint8) * 255

            output_images.append(image)
            output_images.append(mask)

        return output_images, output_masks

    def _update_image(
        self,
        image: np.ndarray,
        x1: np.ndarray,
        x2: np.ndarray,
        x1_object_mask: np.ndarray,
        x2_object_mask: np.ndarray,
        factor: float,
    ):
        (
            image_height,
            image_width,
            min_patch_width,
            min_patch_height,
            max_patch_width,
            max_patch_height,
            patch_width,
            patch_height,
        ) = self._compute_size(image)

        patch_coords, patch_mask = self._find_patch(
            x2_object_mask,
            image_height,
            image_width,
            min_patch_width,
            min_patch_height,
            max_patch_width,
            max_patch_height,
            patch_width,
            patch_height,
        )

        patch = x2[patch_coords.ly : patch_coords.ry, patch_coords.lx : patch_coords.rx]
        patch_height, patch_width = patch.shape[:2]

        if self.args.resize:
            patch, patch_mask = self._resize_patch(
                patch,
                patch_mask,
                min_patch_width,
                min_patch_height,
                max_patch_width,
                max_patch_height,
            )
            patch_height, patch_width = patch.shape[:2]

        x2_object_mask = cv2.resize(
            x2_object_mask[patch_coords.ly : patch_coords.ry, patch_coords.lx : patch_coords.rx],
            (patch_width, patch_height),
        )

        if self.args.shift:
            patch_coords = self._shift_patch(x1, patch, patch_mask, x1_object_mask, x2_object_mask)

        patch_mask &= x2_object_mask | x1_object_mask[patch_coords.ly : patch_coords.ry, patch_coords.lx : patch_coords.rx]  # type: ignore
        blended = self._blend(image, x2, x1_object_mask, x2_object_mask, patch, patch_mask, patch_coords, factor)

        return blended

    def _compute_size(self, image: np.ndarray):
        image_height, image_width = image.shape[:2]
        min_patch_width = int(self.args.min_patch_length_rate * image_width)
        min_patch_height = int(self.args.min_patch_length_rate * image_height)
        max_patch_width = int(self.args.max_patch_length_rate * image_width)
        max_patch_height = int(self.args.max_patch_length_rate * image_height)

        if self.args.gamma_params:
            shape, scale, lower_bound = self.args.gamma_params
            patch_width = int(
                np.clip((lower_bound + np.random.gamma(shape, scale)) * image_width, min_patch_width, max_patch_width)
            )
            patch_height = int(
                np.clip(
                    (lower_bound + np.random.gamma(shape, scale)) * image_height, min_patch_height, max_patch_height
                )
            )
        else:
            patch_width = np.random.randint(min_patch_width, max_patch_height)
            patch_height = np.random.randint(min_patch_height, max_patch_height)

        return (
            image_height,
            image_width,
            min_patch_width,
            min_patch_height,
            max_patch_width,
            max_patch_height,
            patch_width,
            patch_height,
        )

    def _find_patch(
        self,
        object_mask,
        image_height,
        image_width,
        min_patch_width,
        min_patch_height,
        max_patch_width,
        max_patch_height,
        patch_width,
        patch_height,
    ):
        is_found = False
        n_attempts = 0

        lx, ly, rx, ry = 0, 0, 0, 0
        mask = np.empty(shape=())

        while not is_found:
            center_x = np.random.randint(min_patch_width, image_width - min_patch_width)
            center_y = np.random.randint(min_patch_height, image_height - min_patch_height)

            lx = np.clip(center_x - patch_width // 2, 0, image_width)
            ly = np.clip(center_y - patch_height // 2, 0, image_height)
            rx = np.clip(lx + patch_width, 0, image_width)
            ry = np.clip(ly + patch_height, 0, image_height)

            mask = np.ones(
                shape=(ry - ly, rx - lx),
                dtype=np.uint8,
            )

            background_area = np.sum(mask & object_mask[ly:ry, lx:rx])
            patch_area = mask.shape[0] * mask.shape[1]
            is_found = background_area / patch_area > self.args.object_overlap_threshold

            n_attempts += 1

            if n_attempts > MAX_ATTEMPTS:
                logger.error("No suitable patch found.")
                raise ValueError
        return Coords(lx, ly, rx, ry), mask

    def _resize_patch(
        self,
        patch: np.ndarray,
        mask: np.ndarray,
        min_patch_width,
        min_patch_height,
        max_patch_width,
        max_patch_height,
    ):
        height, width, channel = patch.shape
        lb, ub = self.args.resize_bounds
        scale = np.clip(np.random.normal(1, 0.5), lb, ub)
        new_height = int(np.clip(scale * height, min_patch_height, max_patch_height))
        new_width = int(np.clip(scale * width, min_patch_width, max_patch_width))

        if channel == 1:  # grayscale
            src = cv2.resize(patch[..., 0], (new_width, new_height))
            src = src[..., None]
        else:
            src = cv2.resize(patch, (new_width, new_height))

        mask = cv2.resize(mask, (new_width, new_height))

        return src, mask

    def _shift_patch(
        self,
        x1: np.ndarray,
        x2_patch: np.ndarray,
        x2_patch_mask: np.ndarray,
        x1_object_mask: np.ndarray,
        x2_object_mask: np.ndarray,
    ):
        lx, ly, rx, ry = 0, 0, 0, 0

        is_found = False
        n_attempts = 0
        x2_patch_height, x2_patch_width, _ = x2_patch.shape
        x1_height, x1_width, _ = x1.shape

        valid_centers = np.array(np.where(x1_object_mask)).transpose()
        valid_centers = valid_centers[
            (valid_centers[:, 0] >= x2_patch_height // 2)
            & (valid_centers[:, 0] < x1_height - x2_patch_height // 2)
            & (valid_centers[:, 1] >= x2_patch_width // 2)
            & (valid_centers[:, 1] < x1_width - x2_patch_width // 2)
        ]
        np.random.shuffle(valid_centers)

        while not is_found:
            center_y, center_x = valid_centers[n_attempts]
            # center_x = np.random.randint(x2_patch_width // 2 + 1, x1_width - x2_patch_width // 2 - 1)
            # center_y = np.random.randint(x2_patch_height // 2 + 1, x1_height - x2_patch_height // 2 - 1)
            lx = center_x - x2_patch_width // 2
            ly = center_y - x2_patch_height // 2
            rx = center_x + (x2_patch_width + 1) // 2
            ry = center_y + (x2_patch_height + 1) // 2

            patch_slice = (slice(ly, ry), slice(lx, rx))

            and_mask = x1_object_mask[patch_slice] & x2_object_mask & x2_patch_mask
            # or_mask = (x1_object_mask[ly:ry, lx:rx] | x2_object_mask) & x2_patch_mask
            is_found = (
                np.sum(x1_object_mask[patch_slice]) / (x2_patch_mask.shape[0] * x2_patch_mask.shape[1])
                > self.args.object_overlap_threshold
            ) and (np.sum(and_mask) / np.sum(x1_object_mask[patch_slice]) > self.args.and_overlap_threshold)
            n_attempts += 1

            if n_attempts > MAX_ATTEMPTS:
                logger.error("No suitable center found. ")
                raise ValueError
        return Coords(lx, ly, rx, ry)

    def _blend(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        x1_object_mask: np.ndarray,
        x2_object_mask: np.ndarray,
        patch: np.ndarray,
        patch_mask: np.ndarray,
        patch_coords: Coords,
        factor: float,
    ):
        blended = x1.copy()
        mask = patch_mask[..., None]

        mode = self.args.mode.lower()
        patch_slice = (slice(patch_coords.ly, patch_coords.ry), slice(patch_coords.lx, patch_coords.rx))

        if mode == "swap":
            blended[patch_slice] -= mask * blended[patch_slice]
            blended[patch_slice] += mask * patch
        elif mode == "uniform":
            blended = blended.astype(np.float32)
            blended[patch_slice] -= factor * mask * blended[patch_slice]
            blended[patch_slice] += factor * mask * patch
            blended = blended.astype(np.uint8)
        elif mode in ["normal_clone", "mixed_clone"]:
            if mode == "normal_clone":
                _mode = cv2.NORMAL_CLONE
            elif mode == "MIXED_CLONE":
                _mode = cv2.MIXED_CLONE
            else:
                raise ValueError
            int_factor = int(factor * 255)
            scaled_mask = int_factor * (patch_mask | ((1 - x2_object_mask) & (1 - x1_object_mask[patch_slice])))
            scaled_mask[0] = 0
            scaled_mask[-1] = 0
            scaled_mask[:, 0] = 0
            scaled_mask[:, -1] = 0

            if np.sum(scaled_mask > 0) < POSITIVE_MASK_THRESHOLD:
                raise ValueError

            center = (
                (patch_coords.lx + patch_coords.rx) // 2,
                (patch_coords.ly + patch_coords.ry) // 2,
            )
            if x1.shape[-1] == 1:
                _x1 = np.concatenate([x1, np.zeros_like(x1), np.zeros_like(x1)], axis=-1)
                _x2 = np.concatenate([x2, np.zeros_like(x2), np.zeros_like(x2)], axis=-1)
                blended = cv2.seamlessClone(_x1, _x2, scaled_mask, center, _mode)
                blended = blended[..., 0]
            else:
                blended = cv2.seamlessClone(x1, x2, scaled_mask, center, _mode)
        else:
            raise ValueError

        return blended
