import os
import subprocess
import numpy as np
from PIL import Image
from first_seminar import ColorTranslator


def compute_psnr(original, reconstructed):
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in dB between two images."""
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def pad_image_to_8x8(img):
    """Pad image with edge values so height and width are multiples of 8."""
    h, w = img.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded, h, w


def mini_jpeg_pipeline(y_channel, quality_factor, translator):
    """
    Full Mini-JPEG encode + decode pipeline on a 2D grayscale/Y channel:
    8x8 blocks -> DCT -> Quantization -> Zigzag -> RLE -> Decode back to image
    """
    padded_img, orig_h, orig_w = pad_image_to_8x8(y_channel)
    pad_h, pad_w = padded_img.shape

    encoded_blocks = []
    total_rle_symbols = 0

    # --- ENCODING ---
    for i in range(0, pad_h, 8):
        for j in range(0, pad_w, 8):
            block = padded_img[i:i+8, j:j+8]

            # 1. Forward DCT + Quantization
            dct_block = translator.DCT(block)
            dct_q = translator.quantization(dct_block, quality_factor)

            # 2. Zigzag scan (flatten the list of diagonals into a 1D list of 64 values)
            zz_diagonals = translator.zigzag(dct_q)
            zz_flat = [val for diag in zz_diagonals for val in diag]

            # 3. Run-Length Encoding (RLE)
            rle = translator.RLE(zz_flat)
            encoded_blocks.append(rle)
            total_rle_symbols += len(rle)

    # --- DECODING ---
    reconstructed_padded = np.zeros_like(padded_img, dtype=np.float64)
    block_idx = 0

    for i in range(0, pad_h, 8):
        for j in range(0, pad_w, 8):
            rle = encoded_blocks[block_idx]
            block_idx += 1

            # 1. RLE Decoding (convert numpy array to Python list for inverse_zigzag)
            zz_recovered = translator.RLE_decoder(rle).tolist()

            # 2. Inverse Zigzag -> 8x8 quantized DCT block
            dct_q_rec = translator.inverse_zigzag(zz_recovered)

            # 3. Dequantize + Inverse DCT
            dct_rec = translator.dequantize(dct_q_rec, quality_factor)
            block_rec = translator.inv_DCT(dct_rec)

            reconstructed_padded[i:i+8, j:j+8] = block_rec

    # Crop back to original dimensions
    reconstructed_img = np.clip(np.round(reconstructed_padded[:orig_h, :orig_w]), 0, 255).astype(np.uint8)
    return reconstructed_img, total_rle_symbols


def ffmpeg_jpeg_baseline(input_gray_path, qscale):
    """
    Compress and decode an image with FFmpeg MJPEG for comparison.
    Note: FFmpeg -q:v (2 to 31) maps inversely to JPEG quality (2 = best, 31 = worst).
    """
    out_jpg = f"ffmpeg_tmp_q{qscale}.jpg"
    cmd = f'ffmpeg -y -loglevel error -i "{input_gray_path}" -q:v {qscale} "{out_jpg}"'
    subprocess.run(cmd, shell=True, check=True)

    file_size_bytes = os.path.getsize(out_jpg)
    decoded_img = np.array(Image.open(out_jpg).convert("L"))
    os.remove(out_jpg)  # Clean up temporary file
    return decoded_img, file_size_bytes


if __name__ == "__main__":
    # 1. Load image and extract Y channel (luminance)
    path = "images/"
    image_path = os.path.join(path, "amazonas.jpg")
    translator = ColorTranslator()

    rgb_img = np.array(Image.open(image_path).convert("RGB"))
    translator.picture = rgb_img
    yuv_img = translator.rgb_to_yuv()
    y_channel = yuv_img[:, :, 0]  # Extract Y channel

    # Save raw Y channel temporarily to compare against FFmpeg fairly
    raw_y_path = "temp_y_channel.png"
    Image.fromarray(y_channel).save(raw_y_path)

    raw_size_bytes = y_channel.size  # 1 byte per pixel (uncompressed)
    print(f"Original Y-channel resolution: {y_channel.shape[1]}x{y_channel.shape[0]}")
    print(f"Uncompressed raw size: {raw_size_bytes} bytes\n")

    # 2. Test 3 Quality Factors (High, Medium, Low)
    # Mapping roughly to FFmpeg -q:v scale (2 = high quality, 10 = medium, 28 = low)
    test_configs = [
        {"qf": 90, "ffmpeg_q": 2},
        {"qf": 50, "ffmpeg_q": 10},
        {"qf": 10, "ffmpeg_q": 28},
    ]

    print(f"{'QF':<5} | {'Mini-JPEG RLE Symbols':<22} | {'Mini-JPEG PSNR':<15} | {'FFmpeg Bytes':<14} | {'FFmpeg PSNR':<12}")
    print("-" * 80)

    for cfg in test_configs:
        qf = cfg["qf"]
        ff_q = cfg["ffmpeg_q"]

        # Run custom Mini-JPEG pipeline
        rec_mini, rle_symbols = mini_jpeg_pipeline(y_channel, qf, translator)
        psnr_mini = compute_psnr(y_channel, rec_mini)

        # Save reconstructed Mini-JPEG image to inspect visual artifacts
        Image.fromarray(rec_mini).save(f"mini_jpeg_qf{qf}.png")

        # Run FFmpeg JPEG baseline
        rec_ff, ff_bytes = ffmpeg_jpeg_baseline(raw_y_path, ff_q)
        psnr_ff = compute_psnr(y_channel, rec_ff)

        print(f"{qf:<5} | {rle_symbols:<22} | {psnr_mini:>11.2f} dB | {ff_bytes:<14} | {psnr_ff:>9.2f} dB")

    # 3. Hardest B/W compression with FFmpeg (-q:v 31 + grayscale filter)
    hardest_bw_out = "ffmpeg_hardest_bw.jpg"
    cmd_hardest = (
        f'ffmpeg -y -loglevel error -i "{image_path}" '
        f'-vf format=gray -q:v 31 "{hardest_bw_out}"'
    )
    subprocess.run(cmd_hardest, shell=True, check=True)

    hardest_bytes = os.path.getsize(hardest_bw_out)
    hardest_img = np.array(Image.open(hardest_bw_out).convert("L"))
    hardest_psnr = compute_psnr(y_channel, hardest_img)

    print("-" * 80)
    print(f"FFmpeg Hardest B/W (-q:v 31): {hardest_bytes} bytes (saved to {hardest_bw_out}) | PSNR: {hardest_psnr:.2f} dB")

    # Clean up temporary raw Y image
    if os.path.exists(raw_y_path):
        os.remove(raw_y_path)


'''
Use FFmpeg to get the b/w version with the hardest compression you can. Comment the results.
'''