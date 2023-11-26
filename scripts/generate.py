from PIL import Image
from mirage_defect.nsa_generator import NsaGenerator, NsaGeneratorArgs


generator = NsaGenerator(NsaGeneratorArgs(mode="uniform", shift=True, resize=True, num_patches=10))

x1 = Image.open("debug/Screw.jpg")
# x1 = Image.open("debug/gyprock-hd-screws-3d-shadow-2000px.jpg")
x2 = Image.open("debug/gyprock-hd-screws-3d-shadow-2000px.jpg")
caption = "screw"
# x1 = Image.open("debug/threaded-flange-octal.jpg")
# x2 = Image.open("debug/20220430131746_img1_49.jpg")
# caption = "flange"

print(x1.size)

generator.generate(x1, x2, caption)
