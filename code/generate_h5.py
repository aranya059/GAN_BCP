import h5py
import os
import numpy as np
from PIL import Image
from skimage.transform import resize

# Define paths
image_source_folder = "./data/ACDC/3g_images"  # Path where images are saved
label_source_folder = "./data/ACDC/3g_images_mask"  # Path where labels are saved
destination_folder = "./data/ACDC/generated_h5/3g_h5"  # Path where h5 files will be saved

# Create the destination directory if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# List all image and label files
image_files = sorted([f for f in os.listdir(image_source_folder) if f.endswith(".png")])
label_files = sorted([f for f in os.listdir(label_source_folder) if f.endswith(".png")])

# Ensure the number of images and labels match
assert len(image_files) == len(label_files), "Number of images and labels do not match!"

# Desired shape for images and labels
desired_shape = (256, 256)

# Function to create a single h5 file for each image and label
def create_h5_file(image_file, label_file, file_counter):
    # Load the image and label, convert to grayscale
    image = Image.open(os.path.join(image_source_folder, image_file)).convert('L')
    label = Image.open(os.path.join(label_source_folder, label_file)).convert('L')

    # Convert to numpy arrays and resize
    image_np = np.array(image).astype(np.float64)
    label_np = np.array(label).astype(np.uint8)

    image_resized = resize(image_np, desired_shape, order=1, preserve_range=True, anti_aliasing=True)
    label_resized = resize(label_np, desired_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

    # Ensure labels have the correct unique values [0, 1, 2, 3]
    label_resized = np.clip(label_resized, 0, 1)

    # Create the h5 file
    h5_filename = os.path.join(destination_folder, f'black_{file_counter}.h5')
    with h5py.File(h5_filename, 'w') as h5f:
        h5f.create_dataset('image', data=image_resized)
        h5f.create_dataset('label', data=label_resized)

# Process each image and label pair
for file_counter, (image_file, label_file) in enumerate(zip(image_files, label_files), start=1):
    create_h5_file(image_file, label_file, file_counter)

print("H5 file creation complete.")
