import cv2
import numpy as np

from .hog_config import blockSize, blockStride, cellSize, nbins, winSize


class Embedder:
    def __init__(self):
        self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.hog.compute(img)        

def create_svm(save_path: str) -> cv2.ml.SVM:
    """Save path should be a yaml file"""
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10000, 1e-8))
    svm.train(data, cv2.ml.ROW_SAMPLE, labels)
    svm.save(save_path)


