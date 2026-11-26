import numpy as np
from first_seminar import ColorTranslator

block_8x8 = np.array([
    [52, 55, 61, 66, 70, 61, 64, 73],
    [63, 59, 55, 90, 109, 85, 69, 72],
    [62, 59, 68, 113, 144, 104, 66, 73],
    [63, 58, 71, 122, 154, 106, 70, 69],
    [67, 61, 68, 104, 126, 88, 68, 70],
    [79, 65, 60, 70, 77, 68, 58, 75],
    [85, 71, 64, 59, 55, 61, 65, 83],
    [87, 79, 69, 68, 65, 76, 78, 94]
], dtype=float)

translator = ColorTranslator()

# Apply DCT
dct_block = translator.DCT(block_8x8)
print("DCT original (redondeada para visualizar):")
print(np.round(dct_block, 1))

# Compare with different Quality Factors (high, medium, low)
for qf in [90, 50, 10]:
    print(f"\n{'='*50}")
    print(f"QUALITY FACTOR = {qf}")
    print(f"{'='*50}")
    
    # Quantize and dequantize
    dct_q = translator.quantization(dct_block, qf)
    dct_deq = translator.dequantize(dct_q, qf)
    
    # Reconstruct block with inverse DCT
    reconstructed = translator.inv_DCT(dct_deq)
    
    # Count how many coefficients turn 0
    zeros_count = np.sum(dct_q == 0)
    mse = np.mean((block_8x8 - reconstructed) ** 2)
    
    print("Matriz cuantizada (DCT_q):")
    print(dct_q.astype(int))
    print(f"\nCoeficientes anulados (ceros): {zeros_count} / 64")
    print(f"Error cuadrático medio (MSE) tras IDCT: {mse:.2f}")
    print("\nBloque reconstruido:")
    print(reconstructed.astype(int))

'''
Show what happens to the high-frequency coefficients when the quality goes down
'''