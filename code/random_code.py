import h5py
import numpy as np


def print_unique_values(h5_file_path):
    # Open the HDF5 file
    with h5py.File(h5_file_path, 'r') as file:
        # Separate lists for image and mask datasets
        image_datasets = []
        mask_datasets = []

        # Identify datasets for images and masks
        for dataset_name in file.keys():
            if 'image' in dataset_name.lower():
                image_datasets.append(dataset_name)
            elif 'mask' in dataset_name.lower() or 'label' in dataset_name.lower():
                mask_datasets.append(dataset_name)

        # Print unique pixel values for images
        print("Unique pixel values for images:")
        for dataset_name in image_datasets:
            dataset = file[dataset_name]
            data = dataset[:]
            unique_values = np.unique(data)
            print(f"{dataset_name}: {unique_values}")

        # Print unique pixel values for masks
        print("\nUnique pixel values for masks:")
        for dataset_name in mask_datasets:
            dataset = file[dataset_name]
            data = dataset[:]
            unique_values = np.unique(data)
            print(f"{dataset_name}: {unique_values}")


# Replace 'your_file.h5' with the path to your HDF5 file
print_unique_values('./data/ACDC/slices/seed0001.h5')
