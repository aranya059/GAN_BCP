import os
import numpy as np
import h5py
from PIL import Image

def save_images_to_multiple_h5(image_folder, label_folder, output_base_folder, num_splits=20):
    # Get the list of image and label files
    image_files = sorted(os.listdir(image_folder))
    label_files = sorted(os.listdir(label_folder))

    # Ensure the number of images matches the number of labels
    assert len(image_files) == len(label_files), "The number of images and labels must match."

    # Calculate the number of images per split
    images_per_split = len(image_files) // num_splits
    remainder = len(image_files) % num_splits

    # Split the files and save each split in a separate HDF5 file
    for split_idx in range(num_splits):
        start_idx = split_idx * images_per_split
        end_idx = start_idx + images_per_split + (1 if split_idx < remainder else 0)

        # Create the output folder for this split
        output_folder = os.path.join(output_base_folder, f'g_data_{split_idx + 1}')
        os.makedirs(output_folder, exist_ok=True)

        # Create the HDF5 file path for this split
        h5_file_path = os.path.join(output_folder, 'mri_norm2.h5')

        # Open the HDF5 file for writing
        with h5py.File(h5_file_path, 'w') as h5file:
            # Create datasets with appropriate shapes
            first_image = np.array(Image.open(os.path.join(image_folder, image_files[start_idx])).convert('L'))
            first_label = np.array(Image.open(os.path.join(label_folder, label_files[start_idx])).convert('L'))

            images_dataset = h5file.create_dataset('image', (end_idx - start_idx, first_image.shape[0], first_image.shape[1]), dtype=np.float32)
            labels_dataset = h5file.create_dataset('label', (end_idx - start_idx, first_label.shape[0], first_label.shape[1]), dtype=np.uint8)

            # Loop through each image-label pair in this split and store them in the HDF5 file
            for i, (image_file, label_file) in enumerate(zip(image_files[start_idx:end_idx], label_files[start_idx:end_idx])):
                # Prepare the paths for the image and label
                image_path = os.path.join(image_folder, image_file)
                label_path = os.path.join(label_folder, label_file)

                # Open and convert the image to grayscale
                image = Image.open(image_path).convert('L')
                label = Image.open(label_path).convert('L')

                # Convert image and label to numpy arrays
                image_array = np.array(image) / 255.0  # Normalize to [0, 1] if needed
                label_array = np.array(label)

                # Store the image and label in the HDF5 datasets
                images_dataset[i, :, :] = image_array
                labels_dataset[i, :, :] = label_array

                print(f"Added {image_file} and {label_file} to {h5_file_path}")

        # Reopen the file to count images and labels
        with h5py.File(h5_file_path, 'r') as h5file:
            num_images = h5file['image'].shape[0]
            num_labels = h5file['label'].shape[0]
            print(f"Total images in {h5_file_path}: {num_images}")
            print(f"Total labels in {h5_file_path}: {num_labels}")

# Example usage
image_folder = './data/LA/8g_images'  # Replace with your image folder path
label_folder = './data/LA/8g_masks'   # Replace with your label folder path
output_base_folder = './data/LA/Generated_data/'  # Replace with your output base folder path

# Ensure the output directory exists
os.makedirs(output_base_folder, exist_ok=True)

# Run the function to save images and labels to multiple HDF5 files
save_images_to_multiple_h5(image_folder, label_folder, output_base_folder, num_splits=20)
