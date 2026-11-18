import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from first_seminar import ColorTranslator

path = 'images/'
img = np.asarray(Image.open(path+"amazonas.jpg"))

resize= ColorTranslator()
resize.resize(8, path+"amazonas.jpg", path)