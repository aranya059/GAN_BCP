import os
import shutil


def copy_h5_files(list_file, source_dir, destination_dir):
    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Read the list of file names from the text file
    with open(list_file, 'r') as file:
        file_names = file.read().splitlines()

    # Iterate over each file name and copy if it exists in the source directory
    for base_name in file_names:
        # Ensure the file has the .h5 extension
        if not base_name.endswith('.h5'):
            file_name = f"{base_name}.h5"
        else:
            file_name = base_name

        # Construct the full file path
        source_file_path = os.path.join(source_dir, file_name)

        if os.path.exists(source_file_path):
            # Construct the destination file path
            destination_file_path = os.path.join(destination_dir, file_name)
            # Copy the file
            shutil.copy(source_file_path, destination_file_path)
            print(f"Copied {file_name} to {destination_dir}")
        else:
            print(f"File {file_name} not found in {source_dir}")

# Example usage
list_file = './data/ACDC/train.list'  # Path to the text file containing the list of file names
source_dir = './data/ACDC/frame_data'  # Path to the folder containing the h5 files
destination_dir = './data/ACDC/train_frame'  # Path to the destination folder

copy_h5_files(list_file, source_dir, destination_dir)
