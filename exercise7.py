'''
logic taken from:
https://www.geeksforgeeks.org/python/run-length-encoding-python/
'''

from first_seminar import ColorTranslator

zigzagged = [1,1,1,6,6,4,7,7,7,7,8,8,5,3,3,3,3,3,3]

rle_encoder = ColorTranslator()
rle = rle_encoder.RLE(zigzagged)

print("RLE output: ", rle)

rle_decoder = ColorTranslator()
zigzagged_recovered = rle_encoder.RLE_decoder(rle)

print("RLE decoder output: ", zigzagged_recovered)

