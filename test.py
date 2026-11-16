import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from first_seminar import ColorTranslator

path = 'images/'
img = np.asarray(Image.open(path+"amazonas.jpg"))

print(img.shape) 

c1 = ColorTranslator(img)
yuv = c1.rgb_to_yuv()
c4 = ColorTranslator(yuv)
yuv_final = c4.clip_to_integer()

c2 = ColorTranslator(yuv)
rgb = c2.yuv_to_rgb()
c3 = ColorTranslator(rgb)
rgb_final = c3.clip_to_integer()


fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(yuv_final[:, :, 0], cmap='gray')
axes[0].set_title("Canal Y (Luminancia)")
axes[0].axis("off")

axes[1].imshow(rgb_final)
axes[1].set_title("RGB Reconstruido")
axes[1].axis("off")

plt.tight_layout()
plt.show()


