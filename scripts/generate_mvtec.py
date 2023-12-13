from pathlib import Path
from PIL import Image
from mirage_defect.mask_generator import UniformMaskGenerator, GroundendSamMaskGenerator
from mirage_defect.nsa_generator import NsaGenerator, NsaGeneratorArgs


DATASET_ROOT = "/dataset/mvtec"
DST_DIR = "/dataset/mvtec_nsa"
N = 3

OBJECT_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]

TEXTURE_CATEGORIES = [
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
]

CATEGORIES = OBJECT_CATEGORIES + TEXTURE_CATEGORIES


for category in CATEGORIES:
    mask_generator = UniformMaskGenerator() if category in TEXTURE_CATEGORIES else GroundendSamMaskGenerator()
    generator = NsaGenerator(NsaGeneratorArgs(mode="uniform", shift=True, resize=True, num_patches=3), mask_generator)

    dir_path = Path(DATASET_ROOT) / category / "train" / "good"

    for path in dir_path.iterdir():
        im = Image.open(path)
        caption = category

        outputs = generator.generate(im, im, caption, n=N)
        for idx, (x, mask) in enumerate(zip(*outputs)):
            dst_image_path = Path(DST_DIR) / category / "images" / f"{path.stem}_{idx}.bmp"
            dst_mask_path = Path(DST_DIR) / category / "masks" / f"{path.stem}_{idx}.bmp"

            dst_image_path.parent.mkdir(exist_ok=True, parents=True)
            dst_mask_path.parent.mkdir(exist_ok=True, parents=True)

            print(dst_image_path, dst_mask_path)

            Image.fromarray(x).save(dst_image_path)
            Image.fromarray(mask).save(dst_mask_path)

    del mask_generator
    del generator
