import typing

import cv2
import numpy as np


class HOGClassifier:
    def __init__(self, svm_path: typing.Optional[None]):
        self.hog = cv2.HOGDescriptor()
        if svm_path is None:
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        else:
            self.hog.setSVMDetector(HOGClassifier.load_svm_path(svm_path))
        self.default_hog_kwargs = {
            "winStride": (8, 8),
            "padding": 16,
            "scale": 1.05,
            "useMeanShiftGrouping": -1
        }

    @staticmethod
    def load_svm_path(svm_path: str):
        # TO-DO: check exact format to save and reload
        # implement after writing the training part
        pass

    def predict(self, img: np.ndarray, crops: typing.List[typing.Tuple[float, float, float, float]], **kwargs):
        fn_kwargs = {}
        for k, v in self.default_hog_kwargs:
            fn_kwargs[k] = kwargs.get(k, v)
        # TO-DO: make use of crops
        (rects, weights) = self.hog.detectMultiScale(img,**fn_kwargs)
