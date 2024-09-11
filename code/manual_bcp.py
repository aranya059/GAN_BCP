import os
import random
from PIL import Image

# Define folder paths
real_image_folder = './data/ACDC/visulizations/real'  # Replace with your path
synthetic_image_folder = './data/ACDC/visulizations/synthetic'  # Replace with your path
mix_real_folder = './data/ACDC/visulizations/mix_real'
mix_synthetic_folder = './data/ACDC/visulizations/mix_synthetic'

# Ensure the output directories exist
os.makedirs(mix_real_folder, exist_ok=True)
os.makedirs(mix_synthetic_folder, exist_ok=True)

# Function to perform random square crop
def random_square_crop(image, crop_size):
    width, height = image.size
    # Ensure the crop size doesn't exceed the image dimensions
    crop_size = min(crop_size, width, height)
    x = random.randint(0, width - crop_size)
    y = random.randint(0, height - crop_size)
    return x, y, crop_size, crop_size

# Function to paste crop (equivalent to the mix operation)
def paste_crop(target_img, source_img, crop_box):
    cropped = source_img.crop((crop_box[0], crop_box[1], crop_box[0] + crop_box[2], crop_box[1] + crop_box[3]))  # Crop from source image
    target_img.paste(cropped, (crop_box[0], crop_box[1]))  # Paste it onto the target image

# Get image file names and ensure matching real and synthetic images
real_image_files = sorted(os.listdir(real_image_folder))
synthetic_image_files = sorted(os.listdir(synthetic_image_folder))

# Ensure the number of real and synthetic images are the same
assert len(real_image_files) == len(synthetic_image_files), "Real and synthetic folders must contain the same number of images."

# Iterate through all real and synthetic image pairs
for real_img_name, synth_img_name in zip(real_image_files, synthetic_image_files):
    # Open real and synthetic images as grayscale
    real_img_path = os.path.join(real_image_folder, real_img_name)
    synth_img_path = os.path.join(synthetic_image_folder, synth_img_name)

    real_img = Image.open(real_img_path).convert('L')  # Convert to single-channel (grayscale)
    synth_img = Image.open(synth_img_path).convert('L')  # Convert to single-channel (grayscale)

    # Set crop size (1/3rd of the smaller dimension)
    crop_size = min(real_img.size) // 2
    crop_box = random_square_crop(real_img, crop_size)

    # Create copies of the real and synthetic images to avoid overwriting
    real_img_copy = real_img.copy()
    synth_img_copy = synth_img.copy()

    # Paste square from real to synthetic
    paste_crop(synth_img_copy, real_img, crop_box)
    # Paste square from synthetic to real
    paste_crop(real_img_copy, synth_img, crop_box)

    # Save the modified images to the respective folders
    mix_real_img_path = os.path.join(mix_real_folder, real_img_name)
    mix_synth_img_path = os.path.join(mix_synthetic_folder, synth_img_name)

    real_img_copy.save(mix_real_img_path)
    synth_img_copy.save(mix_synth_img_path)

    print(f"Processed and saved: {real_img_name} and {synth_img_name}")

print("Bidirectional copy-paste completed and images saved.")
