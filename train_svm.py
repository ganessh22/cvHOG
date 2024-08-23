import os
import typing

import cv2
import numpy as np
from scipy import ndimage
from skimage.exposure import adjust_gamma
from skimage.feature import hog


def get_features_skimage(img: np.ndarray) -> npdarray:
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_features, _ = hog(
        im,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True,
    )
    return hog_features


def get_features(img: np.ndarray) -> np.ndarray:
    # Initialize HOG descriptor with parameters
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor(
        _winSize=(128, 64),  # Size of the detection window
        _blockSize=(16, 16),  # Block size (2 cells of 8x8)
        _blockStride=(8, 8),  # Block stride (overlap of 8 pixels)
        _cellSize=(8, 8),  # Cell size (8x8 pixels)
        _nbins=9,  # Number of orientation bins
    )

    # Compute HOG features
    hog_features = hog.compute(im)

    return hog_features.flatten()


def blur3(img: np.ndarray) -> np.ndarray:
    return ndimage.uniform_filter(img, size=(3, 3, 1))


def blur4(img: np.ndarray) -> np.ndarray:
    return ndimage.uniform_filter(img, size=(4, 4, 1))


def hflip(img: np.ndarray) -> np.ndarray:
    return img[:, ::-1]


def gamma(img: np.ndarray) -> np.ndarray:
    return adjust_gamma(img, gamma=0.5)


def noise(img: np.ndarray) -> np.ndarray:
    return random_noise(img, mean=0.005)


def augment_image(img: np.ndarray, augment_type: str) -> np.ndarray:
    augs = {
        "blur3": blur3,
        "blur4": blur4,
        "hflip": hflip,
        "gamma": gamma,
        "noise": noise,
    }
    assert augment_type in augs
    return augs[augment_type](img)


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
