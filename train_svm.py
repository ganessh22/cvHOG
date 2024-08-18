import os
import typing

import cv2
import numpy as np
from skimage.feature import hog

# Load and preprocess the image
image = io.imread('path_to_your_image.jpg')  # Replace with your image path
gray_image = color.rgb2gray(image)  # Convert to grayscale

# Compute HOG features
hog_features, hog_image = hog(gray_image, 
                               orientations=9, 
                               pixels_per_cell=(8, 8), 
                               cells_per_block=(2, 2), 
                               visualize=True, 
                               feature_vector=True)

def get_features_skimage()

def get_features(img: np.ndarray) -> np.ndarray:
    # Initialize HOG descriptor with parameters
    hog = cv2.HOGDescriptor(
        _winSize=(128, 64),  # Size of the detection window
        _blockSize=(16, 16),  # Block size (2 cells of 8x8)
        _blockStride=(8, 8),  # Block stride (overlap of 8 pixels)
        _cellSize=(8, 8),  # Cell size (8x8 pixels)
        _nbins=9  # Number of orientation bins
    )
    
    # Compute HOG features
    hog_features = hog.compute(img)

    return hog_features.flatten()


def augment_image(img: np.ndarray, augment_type: str) -> np.ndarray:
    pass


def load_and_resize_images(
    input_folder: str, width: int = 64, height: int = 128
) -> typing.List[np.ndarray]:
    images = []
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        # Check if the file is an image (you can add more formats if needed)
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".gif")
        ):
            # Load the image
            image = cv2.imread(file_path)

            if image is not None:
                # Resize the image
                resized_image = cv2.resize(image, (width, height))
                images.append(resized_image)
            else:
                print(f"Warning: Could not load image {filename}")
    return images
