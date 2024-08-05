import cv2
import numpy as np

from hog_config import blockSize, blockStride, cellSize, nbins, winSize
from loader import Data
from utils import absolute_coord_from_normalized


class Embedder:
    def __init__(self):
        self.hog = cv2.HOGDescriptor(
            winSize, blockSize, blockStride, cellSize, nbins
        )

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.hog.compute(img)


def create_svm(save_path: str) -> cv2.ml.SVM:
    """Save path should be a yaml file"""
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setTermCriteria(
        (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10000, 1e-8)
    )
    svm.train(data, cv2.ml.ROW_SAMPLE, labels)
    svm.save(save_path)


def make_embeddings(
    embedder: Embedder,
    frame: np.ndarray,
    crops: typing.List[typing.Tuple[float, float, float, float]],
    gt: typing.List[bool],
) -> typing.Tuple[np.ndarray, np.ndarray]:
    height, width, _ = frame.shape
    all_feats, all_y = [], []
    for ci, gti in zip(crops, gt):
        x, y, w, h = absolute_coord_from_normalized(ci, height, width)
        cropped_frame = frame[y : y + h, x : x + w]
        all_feats.append(embedder(cropped_frame)[None])
        all_y.append(int(gti))
    return np.concatenate(all_feats, axis=0), np.array(y)


def create_data(data_path: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    embed = Embedder()
    dataset = Data(data_path, augment=True)
    all_data_x, all_data_y = [], []
    for data in dataset:
        img = data["frame"]
        crops = data["crops"]
        gt = data["gt"]
        xd, yd = make_embeddings(embed, img, crops, gt)
        all_data_x.append(xd)
        all_data_y.append(yd)
    return np.concatenate(all_data_x, axis=0), np.concatenate(
        all_data_y, axis=0
    )
