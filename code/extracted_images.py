import os
import h5py
from PIL import Image
import numpy as np

# Base directory containing the folders
base_dir = './data/LA_GAN/gan_train_data/4'

# Output directories for images and masks
output_image_folder = './data/LA_GAN/gan_train_data/e_images'
output_mask_folder = './data/LA_GAN/gan_train_data/e_masks'

# Create output directories if they don't exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

# Iterate over all subfolders in the base directory
for subdir, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.h5'):
            h5_file = os.path.join(subdir, file)

            with h5py.File(h5_file, 'r') as h5f:
                # Check for 'image' and 'label' datasets in the .h5 file
                if 'image' in h5f and 'label' in h5f:
                    images = h5f['image'][:]
                    labels = h5f['label'][:]
                    folder_name = os.path.basename(subdir)

                    # Save each image and mask
                    for i in range(len(images)):
                        # Convert image to 8-bit grayscale and save
                        img_data = np.clip(images[i] * 255, 0, 255).astype('uint8')
                        img = Image.fromarray(img_data)
                        img.save(os.path.join(output_image_folder, f'{folder_name}_image_{i}.png'))

                        # Convert mask to 8-bit grayscale and save
                        lbl_data = np.clip(labels[i] * 255, 0, 255).astype('uint8')
                        lbl = Image.fromarray(lbl_data)
                        lbl.save(os.path.join(output_mask_folder, f'{folder_name}_mask_{i}.png'))

                        print(f'Extracted {folder_name}_image_{i}.png and {folder_name}_mask_{i}.png')
                else:
                    print(f'Datasets "image" or "label" not found in {h5_file}')

print("Extraction completed!")
