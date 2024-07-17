import os
import numpy as np
import h5py
from PIL import Image

def load_images_to_h5(folder_path, h5_file_path):
    # Prepare lists to hold image and label data
    images = []
    labels = []

    # Read each image and label file
    num_images = 250  # Adjust based on your actual number of images
    for i in range(num_images):
        image_path = os.path.join(folder_path, f'generated_image_{i}.png')
        label_path = os.path.join(folder_path, f'generated_label_{i}.png')

        # Open and convert the image to grayscale
        image = Image.open(image_path).convert('L')
        label = Image.open(label_path).convert('L')

        # Convert image and label to numpy arrays and normalize if necessary
        image_array = np.array(image) / 255.0  # Normalize to [0, 1] if needed
        label_array = np.array(label)

        images.append(image_array)
        labels.append(label_array)

    # Convert lists to numpy arrays
    images_np = np.array(images, dtype=np.float64)
    labels_np = np.array(labels, dtype=np.uint8)

    # Save to HDF5 file
    with h5py.File(h5_file_path, 'w') as h5file:
        h5file.create_dataset('image', data=images_np)
        h5file.create_dataset('label', data=labels_np)
        print(f"Data saved to {h5_file_path}")

# Example usage
folder_path = './generated_images_and_labels_epoch_100'
h5_file_path = './generated_data_epoch_500.h5'
load_images_to_h5(folder_path, h5_file_path)
