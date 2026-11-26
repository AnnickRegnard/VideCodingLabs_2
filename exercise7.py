'''
logic taken from:
https://www.geeksforgeeks.org/python/run-length-encoding-python/
https://www.geeksforgeeks.org/dsa/huffman-coding-in-python/ 
'''

from first_seminar import ColorTranslator, build_huffman_tree, generate_huffman_codes

zigzagged = [1,1,1,6,6,4,7,7,7,7,8,8,5,3,3,3,3,3,3]

rle_encoder = ColorTranslator()
rle = rle_encoder.RLE(zigzagged)

print("RLE output: ", rle)

rle_decoder = ColorTranslator()
zigzagged_recovered = rle_encoder.RLE_decoder(rle)

print("RLE decoder output: ", zigzagged_recovered)

values, freq = rle_encoder.RLE_frequencies(rle)
print("\nValores únicos:", values)
print("Frecuencias:", freq)

root = build_huffman_tree(values, freq)
huffman_codes = generate_huffman_codes(root)

print("\nHuffman codes generated:")
for val, code in huffman_codes.items():
    print(f"  Value {val} -> {code}")

encoded_bitstream = "".join(huffman_codes[val] for val in zigzagged)
print("\nCoded sequence (bits):", encoded_bitstream)

