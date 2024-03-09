import logging
import os

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Configure logging
logging.basicConfig(level=logging.INFO, filename='image_compression.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_ssim(image_a, image_b):
    """Calculate the Structural Similarity Index (SSIM) between two images."""
    # Determine the minimum dimension of the images to set an appropriate win_size
    min_dimension = min(image_a.shape[0], image_a.shape[1], image_b.shape[0], image_b.shape[1])
    win_size = min(7, min_dimension // 2 * 2 + 1)  # Ensure win_size is odd and less than the smallest dimension

    print(f"Image A dimensions: {image_a.shape}")
    print(f"Image B dimensions: {image_b.shape}")
    print(f"Using win_size: {win_size}")

    ssim_index = ssim(image_a, image_b, multichannel=True, win_size=win_size, channel_axis=-1)
    return ssim_index


def compress_image_flexible(source_path, target_path, max_size_mb=2, quality_threshold=50, ssim_threshold=0.95,
                            size_tolerance=0.1, summary_filename='compression_summary.txt'):
    original_img = Image.open(source_path).convert('RGB')
    original_img_array = np.array(original_img)

    original_size_bytes = os.path.getsize(source_path)
    max_size_bytes = max_size_mb * 1024 * 1024
    size_tolerance_bytes = max_size_bytes * size_tolerance

    low, high = quality_threshold, 95
    optimal_quality = high
    final_ssim = 0

    while low <= high:
        mid_quality = (low + high) // 2
        original_img.save(target_path, quality=mid_quality, optimize=True)
        compressed_img = Image.open(target_path).convert('RGB')
        compressed_img_array = np.array(compressed_img)
        current_ssim = calculate_ssim(original_img_array, compressed_img_array)

        if os.path.getsize(target_path) <= (max_size_bytes + size_tolerance_bytes):
            if current_ssim >= ssim_threshold:
                optimal_quality = mid_quality
                final_ssim = current_ssim
                high = mid_quality - 1
            else:
                low = mid_quality + 1
        else:
            low = mid_quality + 1

    final_size_bytes = os.path.getsize(target_path)
    size_change = final_size_bytes - original_size_bytes

    # Log and write summary
    log_and_summary = f"Processed {source_path}: Original size = {original_size_bytes} bytes, Final size = {final_size_bytes} bytes, Size change = {size_change} bytes, Final SSIM = {final_ssim}, Quality = {optimal_quality}\n"
    logging.info(log_and_summary)

    # Append summary to a file
    with open(summary_filename, 'a') as summary_file:
        summary_file.write(log_and_summary)

    # Print summary to console
    print(log_and_summary)


# Example usage
source_path = "source1.jpg"
target_path = "target1.jpg"
compress_image_flexible(source_path, target_path)
