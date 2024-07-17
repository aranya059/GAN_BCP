import h5py
import numpy as np
import matplotlib.pyplot as plt

def check_dataset_dtype(file_path):
    with h5py.File(file_path, 'r') as file:
        if 'image' in file:
            image_dataset = file['image']
            # Print the data type of the dataset
            print(f"Data type of 'image' dataset: {image_dataset.dtype}")
        else:
            print("Dataset 'image' not found in the file.")

        if 'label' in file:
            label_dataset = file['label']
            # Print the data type of the dataset
            print(f"Data type of 'label' dataset: {label_dataset.dtype}")
        else:
            print("Dataset 'label' not found in the file.")

def explore_h5_file(file_path):
    with h5py.File(file_path, 'r') as file:
        # Check if 'image' and 'label' datasets exist in the file
        if 'image' in file and 'label' in file:
            images = file['image']
            labels = file['label']

            # Print the total number of images and labels
            print(f"Total images: {images.shape[0]}")
            print(f"Total labels: {labels.shape[0]}")

            # Verify the first dimension matches if needed
            if images.shape[0] != labels.shape[0]:
                print("Warning: The number of images and labels do not match!")

        else:
            print("The specified datasets 'image' and 'label' do not exist in the file.")


def display_images_and_labels_with_overlap(file_path, start_index, end_index):
    with h5py.File(file_path, 'r') as file:
        # Check if the specified range is valid
        total_images = file['image'].shape[0]
        if start_index < 0 or end_index > total_images or start_index >= end_index:
            raise ValueError("Invalid start or end index")

        # Load specific range of images and labels
        images = file['image'][start_index:end_index]
        labels = file['label'][start_index:end_index]

        # Calculate number of images to display
        num_to_display = end_index - start_index

        # Create figure with subplots for each image and its label
        fig, axes = plt.subplots(nrows=num_to_display, ncols=3, figsize=(15, num_to_display * 5))

        if num_to_display == 1:
            axes = np.expand_dims(axes, axis=0)  # Adjust axes array shape for consistent indexing

        for i in range(num_to_display):
            # Original image
            ax = axes[i, 0]
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')
            ax.set_title(f'Original Image {start_index + i + 1}')

            # Label image
            ax = axes[i, 1]
            ax.imshow(labels[i], cmap='gray')  # Change alpha for transparency
            ax.axis('off')
            ax.set_title(f'Label Image {start_index + i + 1}')

            # Overlaid image
            ax = axes[i, 2]
            ax.imshow(images[i], cmap='gray')
            ax.imshow(labels[i], cmap='gray', alpha=0.6)  # Overlay label with transparency
            ax.axis('off')
            ax.set_title(f'Overlay {start_index + i + 1}')

        plt.tight_layout()
        plt.show()



# Specify the path to your HDF5 file
file_path = './data/2018LA_Seg_Training Set/0RZDK210BSMWAA6467LU/mri_norm2.h5'
#file_path = './data/ACDC/frame_data/patient001_frame01.h5'
explore_h5_file(file_path)
check_dataset_dtype(file_path)
display_images_and_labels_with_overlap(file_path, start_index=0, end_index=4)  # Display images and labels from index 10 to 20

