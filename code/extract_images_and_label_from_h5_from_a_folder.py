import os
import h5py
import numpy as np
import imageio

def normalize_to_uint8(data):
    """
    Normalize the data to the range [0, 255] and convert to uint8.
    """
    data_min = data.min()
    data_max = data.max()

    # Handle case where data_min == data_max to avoid division by zero
    if data_max - data_min == 0:
        return np.zeros(data.shape, dtype=np.uint8)

    normalized_data = (data - data_min) / (data_max - data_min) * 255
    return normalized_data.astype(np.uint8)

def extract_images_and_labels(source_dir, image_output_dir, label_output_dir, frame_list_file):
    # Ensure the output directories exist
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)
    if not os.path.exists(label_output_dir):
        os.makedirs(label_output_dir)

    # Read the list of frames from list.txt
    with open(frame_list_file, 'r') as f:
        frame_list = [line.strip() for line in f.readlines()]

    # List all h5 files in the source directory
    h5_files = [f for f in os.listdir(source_dir) if f.endswith('.h5') and os.path.splitext(f)[0] in frame_list]

    if not h5_files:
        print("No matching .h5 files found in the source directory.")
        return

    for h5_file in h5_files:
        print(f"Processing file: {h5_file}")

        # Open the h5 file
        h5_path = os.path.join(source_dir, h5_file)
        with h5py.File(h5_path, 'r') as f:
            # Ensure 'image' and 'label' datasets exist
            if 'image' not in f or 'label' not in f:
                print(f"'image' or 'label' datasets not found in {h5_file}.")
                continue

            image = f['image'][:]
            label = f['label'][:]

            # Normalize to uint8
            image_uint8 = normalize_to_uint8(image)
            label_uint8 = normalize_to_uint8(label)

            # Ensure image and label are 2D
            if image_uint8.ndim != 2 or label_uint8.ndim != 2:
                print(f"Image or label in {h5_file} is not 2D. Skipping file.")
                continue

            # Define the output paths
            base_filename = os.path.splitext(h5_file)[0]
            image_output_path = os.path.join(image_output_dir, f"{base_filename}_image.png")
            label_output_path = os.path.join(label_output_dir, f"{base_filename}_label.png")

            # Save the image and label using imageio
            imageio.imwrite(image_output_path, image_uint8)
            imageio.imwrite(label_output_path, label_uint8)

            print(f"Saved image to {image_output_path} and label to {label_output_path}")

# Example usage
source_dir = './data/ACDC/slices_original'  # Directory containing the h5 files
image_output_dir = './data/ACDC/3e_images'  # Directory where extracted images will be saved
label_output_dir = './data/ACDC/3e_masks'  # Directory where extracted labels will be saved
frame_list_file = './data/ACDC/list.txt'  # File containing the list of frame names to process

extract_images_and_labels(source_dir, image_output_dir, label_output_dir, frame_list_file)
