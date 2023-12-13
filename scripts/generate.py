from PIL import Image
from mirage_defect.mask_generator import GroundendSamMaskGenerator, UniformMaskGenerator
from mirage_defect.nsa_generator import NsaGenerator, NsaGeneratorArgs

# mask_generator = GroundendSamMaskGenerator()
mask_generator = UniformMaskGenerator()
generator = NsaGenerator(NsaGeneratorArgs(mode="uniform", shift=True, resize=True, num_patches=10), mask_generator)

x1 = Image.open("debug/SLV01191107022929007.bmp")
# x1 = Image.open("debug/gyprock-hd-screws-3d-shadow-2000px.jpg")
x2 = Image.open("debug/SLV01191107022929007.bmp")
caption = "chip"
# x1 = Image.open("debug/threaded-flange-octal.jpg")
# x2 = Image.open("debug/20220430131746_img1_49.jpg")
# caption = "flange"

print(x1.size)

x, mask = generator.generate(x1, x2, caption)

Image.fromarray(x).save("debug/output.png")
