import cv2
import numpy as np
from skimage.feature import hog

# from ..train_svm import get_features, get_features_skimage


def get_features_skimage(img: np.ndarray) -> np.ndarray:
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



def test_matching_sklearn_cv2():
    im = np.random.randint(0, 255, (64, 128, 3), dtype=np.uint8)
    cv = get_features(im)
    sk = get_features_skimage(im)
    import pdb; pdb.set_trace()
    np.testing.assert_array_almost_equal(cv, sk, 1e-2)

test_matching_sklearn_cv2()