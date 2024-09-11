import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# Function to extract myocardium voxels from images
def extract_myocardium_voxels(image_folder):
    voxel_values = []
    for img_file in sorted(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, img_file)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_array = np.array(img)

        # Extract voxels with intensity representing myocardium (assuming a specific intensity range for myocardium)
        myocardium_voxels = img_array[
            (img_array > 50) & (img_array < 200)]  # Assuming myocardium has intensity between 50-200
        voxel_values.extend(myocardium_voxels)

        print(f"Extracted {len(myocardium_voxels)} myocardium voxels from {img_file}")

    return voxel_values

# Define folder paths
real_image_folder = './data/ACDC/visulizations/real'
synthetic_image_folder = './data/ACDC/visulizations/synthetic'
mix_real_folder = './data/ACDC/visulizations/mix_real'
mix_synthetic_folder = './data/ACDC/visulizations/mix_synthetic'

# Extract myocardium voxels
real_voxels = extract_myocardium_voxels(real_image_folder)
synthetic_voxels = extract_myocardium_voxels(synthetic_image_folder)
mix_real_voxels = extract_myocardium_voxels(mix_real_folder)
mix_synthetic_voxels = extract_myocardium_voxels(mix_synthetic_folder)

# Print total voxels extracted for each set
print(f"Total real voxels: {len(real_voxels)}")
print(f"Total synthetic voxels: {len(synthetic_voxels)}")
print(f"Total mix_real voxels: {len(mix_real_voxels)}")
print(f"Total mix_synthetic voxels: {len(mix_synthetic_voxels)}")

# Convert integer voxel data to float for KDE
real_voxels_float = np.array(real_voxels, dtype=np.float32)
synthetic_voxels_float = np.array(synthetic_voxels, dtype=np.float32)
mix_real_voxels_float = np.array(mix_real_voxels, dtype=np.float32)
mix_synthetic_voxels_float = np.array(mix_synthetic_voxels, dtype=np.float32)

# First plot: KDE for real vs synthetic images
plt.figure(figsize=(10, 6))
sns.kdeplot(real_voxels_float, label='Real', color='blue', bw_adjust=0.8)
sns.kdeplot(synthetic_voxels_float, label='Synthetic', color='red', bw_adjust=0.8)
plt.title('KDE of Myocardium Voxels: Real vs Synthetic Images')
plt.xlabel('Voxel Intensity')
plt.ylabel('Density')
plt.legend()

# Save the plot as PNG
plt.savefig('real_vs_synthetic_kde.png')

# Show the plot
plt.show()

# Second plot: KDE for mix_real vs mix_synthetic images
plt.figure(figsize=(10, 6))
sns.kdeplot(mix_real_voxels_float, label='Mix Real', color='green', bw_adjust=0.8)
sns.kdeplot(mix_synthetic_voxels_float, label='Mix Synthetic', color='orange', bw_adjust=0.8)
plt.title('KDE of Myocardium Voxels: Mix Real vs Mix Synthetic Images')
plt.xlabel('Voxel Intensity')
plt.ylabel('Density')
plt.legend()

# Save the plot as PNG
plt.savefig('mix_real_vs_mix_synthetic_kde.png')

# Show the plot
plt.show()
