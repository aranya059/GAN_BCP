import os
import h5py

# Path to the text file containing the list of folders
list_file = './data/train.list'

# Base directory containing the folders listed in list_file
base_dir = './data/2018LA_Seg_Training Set'

# Read the list of folders
with open(list_file, 'r') as f:
    folders = [line.strip() for line in f.readlines()]

# Initialize a counter for the total number of images
total_images = 0

# Process the first 8 folders
for folder in folders[:8]:
    folder_path = os.path.join(base_dir, folder)

    # Find the .h5 file in the folder
    h5_file = None
    for file in os.listdir(folder_path):
        if file.endswith('.h5'):
            h5_file = os.path.join(folder_path, file)
            break

    if h5_file:
        with h5py.File(h5_file, 'r') as h5f:
            # Assuming the dataset for images is named 'image'
            if 'image' in h5f:
                num_images = h5f['image'].shape[0]
                print(f'Folder: {folder}, Number of images: {num_images}')
                total_images += num_images
            else:
                print(f'No "image" dataset found in {h5_file}')
    else:
        print(f'No .h5 file found in {folder_path}')

# Print the total number of images across the first 8 folders
print(f'Total number of images across the first 8 folders: {total_images}')
