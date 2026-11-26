import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from first_seminar import ColorTranslator


path = 'images/'
img = np.asarray(Image.open(path+"amazonas.jpg"))

print(img.shape) 

c1 = ColorTranslator(img)
yuv = c1.rgb_to_yuv()


c2 = ColorTranslator(yuv)
rgb = c2.yuv_to_rgb()

print(img[:, :, :][100][100])
print(rgb[:, :, :][100][100])


fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(yuv[:, :, 0], cmap='gray')
axes[0].set_title("Canal Y (Luminancia)")
axes[0].axis("off")

axes[1].imshow(rgb)
axes[1].set_title("RGB Reconstruido")
axes[1].axis("off")

plt.tight_layout()
plt.show()


'''
Round trip: RGB → YUV → RGB. Do you get exactly the same values? If not, why not?
'''

