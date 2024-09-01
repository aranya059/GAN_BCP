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

def extract_images_and_labels(source_dir, image_output_dir, label_output_dir):
    # Ensure the output directories exist
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)
    if not os.path.exists(label_output_dir):
        os.makedirs(label_output_dir)

    # List all h5 files in the source directory
    h5_files = [f for f in os.listdir(source_dir) if f.endswith('.h5')]

    if not h5_files:
        print("No .h5 files found in the source directory.")
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

            images = f['image'][:]
            labels = f['label'][:]

            # Verify the number of images and labels match
            if images.shape[0] != labels.shape[0]:
                print(f"Number of images and labels do not match in {h5_file}. Skipping file.")
                continue

            # Extract and save each image and label
            for i in range(images.shape[0]):
                image = images[i]
                label = labels[i]

                # Normalize to uint8
                image_uint8 = normalize_to_uint8(image)
                label_uint8 = normalize_to_uint8(label)

                # Define the output paths
                base_filename = f"{os.path.splitext(h5_file)[0]}_{i}"
                image_output_path = os.path.join(image_output_dir, f"{base_filename}_image.png")
                label_output_path = os.path.join(label_output_dir, f"{base_filename}_label.png")

                # Save the image and label using imageio
                imageio.imwrite(image_output_path, image_uint8)
                imageio.imwrite(label_output_path, label_uint8)

                print(f"Saved image {i} to {image_output_path} and label {i} to {label_output_path}")

# Example usage
source_dir = './data/ACDC/train_frame'  # Directory containing the h5 files
image_output_dir = './data/ACDC/e_images'  # Directory where extracted images will be saved
label_output_dir = './data/ACDC/e_masks'  # Directory where extracted labels will be saved

extract_images_and_labels(source_dir, image_output_dir, label_output_dir)
