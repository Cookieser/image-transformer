import os
from PIL import Image
import matplotlib.pyplot as plt
import math

# degree  method  image
# sample substract compare l1 l2 computition


def plot_degrees_in_same_transformer(image_dir, image_name, dataset_name, transformer_name, save_dir):
    # Initialize a dictionary to store images for different contrast levels
    dirs = [d for d in os.listdir(image_dir) if d.startswith(f"{dataset_name}_{transformer_name}")]
    dirs.sort(key=lambda x: float(x.split('_')[-1]))

    total_images = len(dirs) + 1

    cols = min(4, total_images)  # Set max 4 images per row
    rows = math.ceil(total_images / cols)

    plt.figure(figsize=(cols * 5, rows * 5))  # Adjust figure size based on number of rows and columns
    plt.suptitle(f"{transformer_name} for {dataset_name}:{image_name}", fontsize=16)

    original_img_path = os.path.join(image_dir, dataset_name, image_name.replace(".png", ".jpg"))
    if os.path.exists(original_img_path):
        img = Image.open(original_img_path)
        print(f"Looking for image at: {original_img_path}")
        plt.subplot(rows, cols, 1)  # First subplot for original image
        plt.imshow(img)
        plt.title('Original')
        plt.axis('off')

    for i, transform_dir in enumerate(dirs):
        # Full path to the image in the current contrast directory
        img_path = os.path.join(image_dir, transform_dir, image_name)

        if os.path.exists(img_path):
            # Load the image
            img = Image.open(img_path)
            print(f"Looking for image at: {img_path}")

            plt.subplot(rows, cols, i + 2)
            plt.imshow(img)
            plt.title(f"{transform_dir.split('_')[-1]}")
            plt.axis('off')

    # Save the plot in the specified folder
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Create the directory if it doesn't exist
    save_path = os.path.join(save_dir, f"{dataset_name}_{transformer_name}_{image_name}")
    plt.savefig(save_path)
    print(f"Plot saved at: {save_path}")
    plt.show()
    plt.close()  # Close the plot to avoid displaying it



