import numpy as np
import subprocess


'''
Hay que acordarse de añadir comentarios y responder las preguntas al final.
Y hacer los bonuses si sobra tiempo.
'''


class ColorTranslator:
    def __init__(self, picture = None):
        self.picture = picture

    def rgb_to_yuv(self):
        #Extract color channels
        R = self.picture[:, :, 0].astype(float)
        G = self.picture[:, :, 1].astype(float)
        B = self.picture[:, :, 2].astype(float)


        #Calculate YUB given RGB
        Y = 0.257 * R + 0.504 * G + 0.098 * B + 16
        U = -0.148 * R - 0.291 * G + 0.439 * B + 128
        V = 0.439 * R - 0.368 * G - 0.071 * B + 128

        picture_yuv = np.dstack((Y, U,V))

        return np.clip(np.round(picture_yuv), 0, 255).astype(np.uint8)

    def yuv_to_rgb(self):
        #Extract color channels
        Y = self.picture[:, :, 0].astype(float)
        U = self.picture[:, :, 1].astype(float)
        V = self.picture[:, :, 2].astype(float)


        #Calculate YUB given RGB
        B = 1.164 * (Y - 16) + 2.018 * (U - 128)
        G = 1.164 * (Y - 16) - 0.813 * (V - 128) - 0.391 * (U - 128)
        R = 1.164 * (Y - 16) + 1.596 * (V - 128) 

        picture_rgb = np.dstack((R, G, B))

        return np.clip(np.round(picture_rgb), 0, 255).astype(np.uint8)

    def resize(self, scale, img_path, exit_path):
        txt = f'ffmpeg -y -i {img_path} -vf "scale=iw/{scale}:ih/{scale}" "{exit_path}/image_scaled_by_{scale}.jpg"'

        subprocess.run(txt, check = True, shell = True)

    def zigzag(self, matrix):
        rows, columns = matrix.shape
        solution = [[] for i in range(rows+columns-1)]
        for i in range(rows):
            for j in range(columns):
                sum = i+j
                if(sum % 2 == 0):
                    # add at beginning
                    solution[sum].insert(0, matrix[i][j])
                else:
                    # add at end of the list
                    solution[sum].append(matrix[i][j])
        return solution

    def inverse_zigzag(self, zigzag):
        matrix = np.zeros((8,8))
        # keep track of what diagonal we are on and in what direction we are going
        direction = 1 # 0 is going down a diagonal and 1 is up a diagonal
        diagonal_num = 0
        
        while len(zigzag) != 0: # while the zigzag array is not empty
            # fill the matrix values of the current diagonal
            for i in range(8):
                for j in range(8):
                    if i+j == diagonal_num: # if the diagonal corresponds to the one we are currently on
                        if direction == 0: # downwards
                            matrix[i][j] = zigzag.pop(0) # fill from the top of the diagonal
                        elif direction == 1: # upwards
                            matrix[j][i] = zigzag.pop(0) # start filling from the bottom of the diagonal

            diagonal_num += 1 # move to the next diagonal
            direction = 1 - direction # change directions
            
        return matrix

    def RLE(self, zigzag):
        rle = [] # output will be each value followed by its count e.g. [3, 2, 1, 4] means 331111
        count = 1 # keep track of how many times a value is repeated

        for i in range(1, len(zigzag)):
            if zigzag[i] == zigzag[i - 1]: # if there is a repetition, increment the count
                count += 1
            else: # otherwise, add the value with its count to the output array and set the count back to 1
                rle.append(zigzag[i-1])
                rle.append(count)
                count = 1

        rle.append(zigzag[-1])
        rle.append(count)

        return rle

    def RLE_decoder(self, rle):
        decoded_rle = np.array([])
        for i in range(1, len(rle), 2): # increase by 2 so counts are not confused with values (since it goes: value, count, value, count...)
            value = rle[i-1]
            count = rle[i]
            values = [value]*count 
            decoded_rle = np.append(decoded_rle, values)

        return decoded_rle
