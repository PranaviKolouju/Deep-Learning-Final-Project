import numpy as np
from PIL import Image

# Path to the .npz file
file_path = "../data/Kamitani/npz/images_256.npz"

# Load the .npz file
data = np.load(file_path)

# Extract the key that contains the images (replace 'test_images' with the correct key if needed)
images = data['test_images']

# Ensure the data is valid for saving images
if len(images) < 3:
    raise ValueError("The dataset contains fewer than 3 images.")

# Save the first 3 images as PNG
output_folder = "./"
for i in range(3):
    # Convert image data to uint8 (if not already) and save it
    img = Image.fromarray(images[i].astype(np.uint8))
    img.save(f"{output_folder}image_{i+1}.png")

print("First 3 images saved as PNG.")
