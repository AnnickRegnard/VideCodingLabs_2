import numpy as np

class ColorTranslator:
    def __init__(self, picture):
        self.picture = picture

    def rgb_to_yuv(self):
        #Extract color channels
        R = self.picture[:, :, 0]
        G = self.picture[:, :, 1]
        B = self.picture[:, :, 2]


        #Calculate YUB given RGB
        Y = 0.257 * R + 0.504 * G + 0.098 * B + 16
        U = -0.148 * R - 0.291 * G + 0.439 * B + 128
        V = 0.439 * R - 0.368 * G - 0.071 * B + 128

        picture_yuv = np.dstack((Y, U,V))

        return picture_yuv

    def yuv_to_rgb(self):
        #Extract color channels
        Y = self.picture[:, :, 0]
        U = self.picture[:, :, 1]
        V = self.picture[:, :, 2]


        #Calculate YUB given RGB
        B = 1.164 * (Y - 16) + 2.018 * (U - 128)
        G = 1.164 * (Y - 16) - 0.813 * (V - 128) - 0.391 * (U - 128)
        R = 1.164 * (Y - 16) + 1.596 * (V - 128) 

        picture_rgb = np.dstack((R, G, B))

        return picture_rgb

    def clip_to_integer(self):
        return np.clip(self.picture, 0, 255).astype(np.uint8)



        