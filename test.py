import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

MODEL_TYPE = "vit_b"
cuda = "cuda:0" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[MODEL_TYPE](checkpoint="sam_vit_b.pth")
sam.to(cuda)
mask_generator = SamAutomaticMaskGenerator(sam)


def randomize_sigma():
    _min, _max = 50, 100
    # pick a random sigma between 50 and 100
    sigma = np.random.randint(_min, _max)
    return 10000


def add_gaussian_noise_and_save(
    image_base64, mean=0, sigma=1000, save_path="image.png"
):
    # Decode base64 image to NumPy array
    image_bytes = base64.b64decode(image_base64)
    image_np = np.array(Image.open(BytesIO(image_bytes)))

    masks = mask_generator.generate(image_np)
    for mask in masks:
        # bool type
        segment = mask["segmentation"]

        # Convert the image to float for noise addition
        image_float = image_np.astype(np.float64)

        # Add Gaussian noise only where the mask is True
        noise = np.random.normal(mean, sigma, image_float.shape)
        image_float[segment] += noise[segment]

        # Clip the pixel values to be in the valid range [0, 255]
        image_float = np.clip(image_float, 0, 255)

        # Convert the image back to uint8
        image_np = image_float.astype(np.uint8)

    # Save the noisy image to the local filesystem
    cv2.imwrite(save_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    return save_path


# Example usage
# Replace 'your_image_base64_data' with the actual base64 encoded image data
with open("base64.txt", "r") as f:
    your_image_base64_data = f.read()
saved_path = add_gaussian_noise_and_save(your_image_base64_data)

# 'saved_path' now contains the path where the noisy image is saved
print(f"Noisy image saved to: {saved_path}")
