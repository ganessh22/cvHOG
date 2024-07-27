import typing

import cv2
import numpy as np

from utils import absolute_coord_from_normalized


class HOGClassifier:
    def __init__(self, svm_path: typing.Optional[None]):
        self.hog = cv2.HOGDescriptor()
        if svm_path is None:
            self.hog.setSVMDetector(
                cv2.HOGDescriptor_getDefaultPeopleDetector()
            )
        else:
            self.hog.setSVMDetector(HOGClassifier.load_svm_path(svm_path))
        self.default_hog_kwargs = {
            "winStride": (8, 8),
            "padding": 16,
            "scale": 1.05,
            "useMeanShiftGrouping": -1,
        }

    @staticmethod
    def load_svm_path(svm_path: str):
        # TO-DO: check exact format to save and reload
        # implement after writing the training part
        pass

    def detect(
        self,
        img: np.ndarray,
        crops: typing.List[typing.Tuple[float, float, float, float]],
        **kwargs
    ) -> typing.List[typing.Tuple[float, float, float, float]]:
        fn_kwargs = {}
        for k, v in self.default_hog_kwargs:
            fn_kwargs[k] = kwargs.get(k, v)

        if len(crops) == 0:
            crops = [(0.0, 0.0, 1.0, 1.0)]
        all_rects = []
        height, width, _ = img.shape
        for c in crops:
            x, y, w, h = absolute_coord_from_normalized(c, height, width)
            img_crop = img[y : y + h, x : x + w]
            rects = self.hog.detectMultiScale(img_crop, **fn_kwargs)
            for r in rects:
                x1, y1, w1, h1 = r
                all_rects.append((x1 + x, y1 + y, w1, h1))
        return all_rects

    def predict(
        self,
        img: np.ndarray,
        crops: typing.List[typing.Tuple[float, float, float, float]] = [],
        visualize: bool = False,
        **kwargs
    ) -> typing.Tuple[bool, typing.Optional[np.ndarray]]:
        rects = self.detect(img, crops, **kwargs)
        if visualize:
            img_annotated = self.visualize(img, rects)
            return len(rects) > 0, img_annotated
        return len(rects) > 0, None

    def visualize(
        self,
        img: np.ndarray,
        rects: typing.List[typing.Tuple[float, float, float, float]],
    ) -> np.ndarray:
        for x, y, w, h in rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return img
