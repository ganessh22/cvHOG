import os
import typing

import cv2
import joblib
import numpy as np
from scipy import ndimage
from skimage.exposure import adjust_gamma
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC


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
    input_folder: str,
    width: int = 64,
    height: int = 128,
    augment: bool = False,
    feature_fn: typing.Optional[typing.Callable] = None,
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
                if augment:
                    for aug in ["blur3", "blur4", "hflip", "gamma", "noise"]:
                        aug_image = augment_image(image, aug)
                        resized_image = cv2.resize(aug_image, (width, height))
                        if feature_fn is not None:
                            feat = feature_fn(resized_image)
                        else:
                            feat = resized_image
                        images.append(feat)
                else:
                    resized_image = cv2.resize(image, (width, height))
                    if feature_fn is not None:
                        feat = feature_fn(resized_image)
                    else:
                        feat = resized_image
                    images.append(feat)
            else:
                print(f"Warning: Could not load image {filename}")
    return images


def load_data(
    folder: str, augment: bool = False
) -> typing.Tuple[np.ndarray, np.ndarray]:
    X_1 = load_and_resize_images(
        os.path.join(folder, "positive"),
        feature_fn=get_features,
        augment=augment,
    )
    X_0 = load_and_resize_images(
        os.path.join(folder, "negative"),
        feature_fn=get_features,
        augment=augment,
    )
    y_1 = np.ones((X_1.shape[0],), dtype=np.float32)
    y_0 = np.zeros((X_0.shape[0],), dtype=np.float32)
    return np.concatenate([X_0, X_1], axis=0), np.concatenate(
        [y_0, y_1], axis=0
    )


def get_train_val_data(
    train_folder: str, val_folder: str
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_X, train_y = load_data(train_folder, True)
    val_X, val_y = load_data(val_folder, False)
    return train_X, train_y, val_X, val_y


def train(train_folder: str, val_folder: str, save_path: str = "model.pkl"):
    X, y, Xt, yt = get_train_val_data(train_folder, val_folder)
    model = LinearSVC(class_weight={0: 1.0, 1: 1000.0}, C=0.001)
    model.fit(X, y)
    yp = model.predict(Xt)
    for n, d in zip(["train", "test"], [(X, y), (Xt, yt)]):
        yp = model.predict(d[0])
        print(n + "\n" + "=" * 5)
        for name, fn in zip(
            ["accuracy", "f1", "precision", "recall", "confusion_matrix"],
            [
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
                confusion_matrix,
            ],
        ):
            print(f"{name} : {fn(d[1], yp)}")
    joblib.dump(model, save_path)
