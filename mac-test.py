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
    # Pick a random sigma between 50 and 100
    sigma = np.random.randint(_min, _max)
    return 1000 

def add_gaussian_noise_and_save(
    image_base64, mean=0, sigma=1000, save_path="image.png"
):
    # Decode base64 image to NumPy array
    image_bytes = base64.b64decode(image_base64)
    image_pil = Image.open(BytesIO(image_bytes))
    image_np = np.array(image_pil)

    # Convert the image to PyTorch tensor and move to GPU
    image_tensor = torch.tensor(image_np).float().to(cuda)

    masks = mask_generator.generate(image_np)  # Assuming this works with NumPy arrays
    for mask in masks:
        segment = mask["segmentation"]

        # Convert the mask to tensor and move to GPU
        segment_tensor = torch.tensor(segment).to(cuda)

        # Add Gaussian noise only where the mask is True
        noise_tensor = torch.normal(mean, sigma, image_tensor.shape).to(cuda)
        image_tensor[segment_tensor] += noise_tensor[segment_tensor]

        # Clip the pixel values to be in the valid range [0, 255]
        image_tensor = torch.clamp(image_tensor, 0, 255)

    # Convert the image back to CPU and NumPy for saving
    image_np = image_tensor.to('cpu').numpy().astype(np.uint8)

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
