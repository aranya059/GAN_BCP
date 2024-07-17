import os
import h5py
from PIL import Image
import numpy as np

def save_images_and_labels(base_folder, output_base_folder):
    # Navigate through the base folder
    for subdir, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith('.h5'):  # Check if the file is an HDF5 file
                filepath = os.path.join(subdir, file)
                # Extract the directory name from the path to use as a unique folder name for outputs
                directory_name = os.path.basename(subdir)
                output_folder = os.path.join(output_base_folder, directory_name)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)  # Create the output folder if it doesn't exist
                print(f"Processing {filepath} and saving in {output_folder}")
                extract_and_save_images_labels(filepath, output_folder)

# def save_images_and_labels(base_folder, output_base_folder):
#     files = [f for f in os.listdir(base_folder) if f.endswith('.h5')]
#
#     # Process each file
#     for file in files:
#         filepath = os.path.join(base_folder, file)
#         # Use the file name without extension as the directory name for outputs
#         directory_name = os.path.splitext(file)[0]
#         output_folder = os.path.join(output_base_folder, directory_name)
#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)
#         print(f"Processing {filepath} and saving in {output_folder}")
#         extract_and_save_images_labels(filepath, output_folder)


def extract_and_save_images_labels(file_path, output_folder):
    with h5py.File(file_path, 'r') as h5file:
        # Check for the required datasets 'image' and 'label'
        if 'image' in h5file and 'label' in h5file:
            images = h5file['image'][:]
            labels = h5file['label'][:]
            # Ensure output directory exists
            os.makedirs(output_folder, exist_ok=True)
            # Save each image and label
            for i in range(len(images)):
                # Scale the image data from 0-1 to 0-255 and convert to uint8
                img_data = np.clip(images[i] * 255, 0, 255).astype('uint8')
                img = Image.fromarray(img_data)
                img.save(os.path.join(output_folder, f'image_{i}.png'))
                # Handle labels with same min and max
                if labels[i].min() == labels[i].max():
                    lbl_data = np.zeros_like(labels[i], dtype=np.uint8)  # Create a black image if all label values are the same
                else:
                    lbl_data = np.interp(labels[i], (labels[i].min(), labels[i].max()), (0, 255)).astype('uint8')

                lbl = Image.fromarray(lbl_data)
                lbl.save(os.path.join(output_folder, f'label_{i}.png'))
                print(f'Saved image_{i}.png and label_{i}.png in {output_folder}')
        else:
            print(f"Required datasets 'image' or 'label' not found in {file_path}")


# Specify the path to the main folder and output base folder
base_folder = './data/2018LA_Seg_Training Set'
output_base_folder = './LA_Extracted_Images_Labels'  # All outputs will be saved here, in subfolders per dataset
save_images_and_labels(base_folder, output_base_folder)
