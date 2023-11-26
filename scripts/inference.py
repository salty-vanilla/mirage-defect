from PIL import Image
from mirage_defect.grounded_sam import GroundedSAM
from groundingdino.util.inference import load_image, annotate


model = GroundedSAM()

image = Image.open("debug/gyprock-hd-screws-3d-shadow-2000px.jpg")
boxes, logits, phrases, masks = model.predict(image, "screw")

for i, mask in enumerate(masks):
    Image.fromarray(mask.astype("uint8") * 255).save(f"debug/{i}.png")
