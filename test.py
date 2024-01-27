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
    # Decode base64 image to cv2.cvtColor
    image = Image.open(BytesIO(base64.b64decode(image_base64)))
    # load np array image into cv2
    image_np = np.array(image)

    masks = mask_generator.generate(image_np)
    for mask in masks:
        print(mask["area"])
        sigma = randomize_sigma()
        bbox = mask["bbox"]
        obj = image_np[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        obj = obj + np.random.normal(mean, sigma, obj.shape)

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
