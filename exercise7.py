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

# 2. Calcular frecuencias desde el RLE
values, freq = rle_encoder.RLE_frequencies(rle)
print("\nValores únicos:", values)
print("Frecuencias:", freq)

# 3. Construir árbol y códigos de Huffman
root = build_huffman_tree(values, freq)
huffman_codes = generate_huffman_codes(root)

print("\nCódigos de Huffman generados:")
for val, code in huffman_codes.items():
    print(f"  Valor {val} -> {code}")

# 4. Codificar la secuencia original usando los códigos de Huffman
encoded_bitstream = "".join(huffman_codes[val] for val in zigzagged)
print("\nSecuencia codificada (bits):", encoded_bitstream)

