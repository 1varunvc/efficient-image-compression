import logging
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
from multiprocessing import Pool, freeze_support
import time

# Configure logging
logging.basicConfig(level=logging.INFO, filename='image_compression.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_ssim(image_a, image_b):
    """Calculate the Structural Similarity Index (SSIM) between two images."""
    min_dimension = min(image_a.shape[0], image_a.shape[1], image_b.shape[0], image_b.shape[1])
    win_size = min(7, min_dimension // 2 * 2 + 1)
    ssim_index = ssim(image_a, image_b, multichannel=True, win_size=win_size, channel_axis=-1)
    return ssim_index

def compress_image(args):
    source_path, target_dir, max_size_mb, quality_threshold, ssim_threshold, size_tolerance, summary_filename = args
    file_name = os.path.basename(source_path)
    target_path = os.path.join(target_dir, file_name)
    
    # Check if the target file already exists, if so, skip processing
    if os.path.exists(target_path):
        logging.info(f"Skipping {source_path}, target file already exists.")
        return
    
    try:
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
            original_img.save(target_path, quality=mid_quality, optimize=True, exif=original_img.info.get('exif'))
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

        log_and_summary = f"Processed {source_path}: Original size = {original_size_bytes} bytes, Final size = {final_size_bytes} bytes, Size change = {size_change} bytes, Final SSIM = {final_ssim}, Quality = {optimal_quality}\n"
        logging.info(log_and_summary)

        with open(summary_filename, 'a') as summary_file:
            summary_file.write(log_and_summary)

        print(log_and_summary)
    
    except Exception as e:
        logging.error(f"Error processing {source_path}: {str(e)}")

def compress_images_in_directory(source_dir, target_dir, max_size_mb=2, quality_threshold=50, ssim_threshold=0.95, size_tolerance=0.1, summary_filename='compression_summary.txt'):
    num_processes = min(os.cpu_count() // 2, 4)  # Limit processes to half the number of CPU cores or 4, whichever is smaller
    pool = Pool(processes=num_processes)  # Initialize multiprocessing Pool
    
    tasks = []  # Store tasks for batching
    for root, dirs, files in os.walk(source_dir):
        relative_path = os.path.relpath(root, source_dir)
        target_root = os.path.join(target_dir, relative_path)
        os.makedirs(target_root, exist_ok=True)
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                source_path = os.path.join(root, file)
                args = (source_path, target_root, max_size_mb, quality_threshold, ssim_threshold, size_tolerance, summary_filename)
                tasks.append(args)
                if len(tasks) == num_processes:  # Batch process if enough tasks
                    pool.map_async(compress_image, tasks)
                    tasks = []  # Reset tasks for the next batch
    
    # Process remaining tasks
    if tasks:
        pool.map_async(compress_image, tasks)

    pool.close()
    pool.join()

if __name__ == '__main__':
    freeze_support()
    source_dir = "source"
    target_dir = "target"
    compress_images_in_directory(source_dir, target_dir)
    
    # Add a delay to ensure all processes have completed before exiting
    time.sleep(5)