import os
import pandas as pd
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from functools import partial
import requests
from PIL import Image
import io

# You will need to have torchvision installed: pip install torchvision
import torch
from torchvision import transforms

# --- 1. Define the Image Transformations ---

class ConvertToRGB(object):
    """Converts a PIL image to RGB format if it's not already."""
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

# The transform pipeline to be applied to each image before saving
transform_pipeline = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)), # Resize to a standard 224x224
])


# --- 2. Download and Transform Function ---

def download_and_transform_image(image_link, savefolder, transform):
    """
    Downloads an image from a URL, applies transformations in memory,
    and saves the processed image.
    """
    if not isinstance(image_link, str):
        return

    try:
        filename = Path(image_link).name
        image_save_path = os.path.join(savefolder, filename)

        if os.path.exists(image_save_path):
            return

        response = requests.get(image_link, timeout=10)
        response.raise_for_status()

        image_bytes = io.BytesIO(response.content)
        with Image.open(image_bytes) as img:
            transformed_img = transform(img)
            transformed_img.save(image_save_path)

    except requests.exceptions.RequestException as ex:
        print(f'Warning: Request failed for - {image_link}\n{ex}')
    except (IOError, Image.UnidentifiedImageError) as ex:
        print(f'Warning: Could not process image - {image_link}\n{ex}')
    except Exception as ex:
        print(f'Warning: An unexpected error occurred for - {image_link}\n{ex}')
    return


# --- 3. Main Multiprocessing Function ---

def process_images_in_parallel(image_links, download_folder, transform):
    """
    Processes a list of image links in parallel using multiprocessing.
    """
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    processing_function = partial(
        download_and_transform_image,
        savefolder=download_folder,
        transform=transform
    )

    # Use a multiprocessing pool to parallelize the downloads.
    # 32 processes is a more stable number than 100 for most systems.
    with multiprocessing.Pool(32) as pool:
        for _ in tqdm(pool.imap_unordered(processing_function, image_links), total=len(image_links), desc="Downloading and Transforming Images"):
            pass
    
    print("Image processing complete.")