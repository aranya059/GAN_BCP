import os
import numpy as np
import h5py
from PIL import Image

def save_images_to_h5(image_folder, label_folder, output_folder):
    # Get the list of image and label files
    image_files = sorted(os.listdir(image_folder))
    label_files = sorted(os.listdir(label_folder))

    # Ensure the number of images matches the number of labels
    assert len(image_files) == len(label_files), "The number of images and labels must match."

    for i, (image_file, label_file) in enumerate(zip(image_files, label_files)):
        # Prepare the paths for the image and label
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, label_file)

        # Open and convert the image to grayscale
        image = Image.open(image_path).convert('L')
        label = Image.open(label_path).convert('L')

        # Convert image and label to numpy arrays
        image_array = np.array(image) / 255.0  # Normalize to [0, 1] if needed
        label_array = np.array(label)

        # Create HDF5 file name
        h5_file_name = f'{os.path.splitext(image_file)[0]}.h5'
        h5_file_path = os.path.join(output_folder, h5_file_name)

        # Save to HDF5 file
        with h5py.File(h5_file_path, 'w') as h5file:
            h5file.create_dataset('image', data=image_array, dtype=np.float32)
            h5file.create_dataset('label', data=label_array, dtype=np.uint8)
            print(f"Data saved to {h5_file_path}")

# Example usage
image_folder = './data/ACDC/g_images'
label_folder = './data/ACDC/g_masks'
output_folder = './data/ACDC/Generated_data/Generated_h5'
os.makedirs(output_folder, exist_ok=True)
save_images_to_h5(image_folder, label_folder, output_folder)
