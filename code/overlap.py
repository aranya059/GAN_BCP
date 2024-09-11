import matplotlib.pyplot as plt
from PIL import Image
import os


# Function to display image and mask overlap in a single plot
def display_images_and_masks(images_dir, masks_dir, start_index, num_images=5, colormap='hot', alpha=0.5):
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Sort to ensure corresponding images and masks are paired correctly
    image_files.sort()
    mask_files.sort()

    # Ensure the start_index is within the bounds
    if start_index < 0 or start_index >= len(image_files):
        print(f"Invalid start_index {start_index}. It should be between 0 and {len(image_files) - 1}.")
        return

    # Adjust num_images if there are not enough images left from the start_index
    num_images = min(num_images, len(image_files) - start_index)

    # Create a figure with subplots
    fig, axs = plt.subplots(num_images, 3, figsize=(18, num_images * 6))

    for i in range(num_images):
        image_path = os.path.join(images_dir, image_files[start_index + i])
        mask_path = os.path.join(masks_dir, mask_files[start_index + i])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Display the original image
        axs[i, 0].imshow(image)
        axs[i, 0].set_title(f'Original Image {start_index + i + 1}')
        axs[i, 0].axis('off')

        # Display the mask
        axs[i, 1].imshow(mask, cmap='gray')
        axs[i, 1].set_title(f'Mask {start_index + i + 1}')
        axs[i, 1].axis('off')

        # Display the mask overlay on the original image
        axs[i, 2].imshow(image)
        axs[i, 2].imshow(mask, cmap=colormap, alpha=alpha)  # Overlay the mask with chosen colormap and transparency
        axs[i, 2].set_title(f'Image {start_index + i + 1} with Mask Overlay')
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    images_dir = "./data/for_figure/fives_quantitative/original_images"
    masks_dir = "./data/for_figure/fives_quantitative/unet_gt_10"
    # images_dir = "./Dataset/resized_images"
    # masks_dir = "./Dataset/resized_masks"
    start_index = 0  # Replace this with the desired start index

    display_images_and_masks(images_dir, masks_dir, start_index, colormap='hot', alpha=0.5)