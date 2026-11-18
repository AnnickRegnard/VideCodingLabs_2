'''
zigzag function from:
https://www.geeksforgeeks.org/python/python-program-for-print-matrix-in-zag-zag-fashion/
'''

from first_seminar import ColorTranslator
import numpy as np

matrix1 = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8],
    [9, 10, 11, 12, 13, 14, 15, 16],
    [17, 18, 19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30, 31, 32],
    [33, 34, 35, 36, 37, 38, 39, 40],
    [41, 42, 43, 44, 45, 46, 47, 48],
    [49, 50, 51, 52, 53, 54, 55, 56],
    [57, 58, 59, 60, 61, 62, 63, 64]
])

zigzag = ColorTranslator()
zigzagged_matrix = zigzag.zigzag(matrix1)

for i in zigzagged_matrix:
    for j in i:
        print(j, end=" ")