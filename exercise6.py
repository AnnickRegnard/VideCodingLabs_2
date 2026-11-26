import numpy as np
from first_seminar import ColorTranslator

# Bloque 8x8 clásico de luminancia (valores de píxeles 0-255)
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

# 1. Aplicar DCT-II
dct_block = translator.DCT(block_8x8)
print("DCT original (redondeada para visualizar):")
print(np.round(dct_block, 1))

# 2. Comparar diferentes Quality Factors (alto, medio, bajo)
for qf in [90, 50, 10]:
    print(f"\n{'='*50}")
    print(f"QUALITY FACTOR = {qf}")
    print(f"{'='*50}")
    
    # Cuantizar y descuantizar
    dct_q = translator.quantization(dct_block, qf)
    dct_deq = translator.dequantize(dct_q, qf)
    
    # Reconstruir bloque con Inverse DCT
    reconstructed = translator.inv_DCT(dct_deq)
    
    # Contar cuántos coeficientes se vuelven 0 (especialmente altas frecuencias)
    zeros_count = np.sum(dct_q == 0)
    mse = np.mean((block_8x8 - reconstructed) ** 2)
    
    print("Matriz cuantizada (DCT_q):")
    print(dct_q.astype(int))
    print(f"\nCoeficientes anulados (ceros): {zeros_count} / 64")
    print(f"Error cuadrático medio (MSE) tras IDCT: {mse:.2f}")
    print("\nBloque reconstruido:")
    print(reconstructed.astype(int))