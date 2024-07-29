import glob
import os
import typing

import cv2
import numpy as np


def get_file_by_ext_from_dir(directory: str, ext: str) -> typing.Optional[str]:
    files = list(glob.glob(os.path.join(directory, f"*.{ext}")))
    if len(files) == 0:
        return None
    return files[0]


def absolute_coord_from_normalized(
    coords: typing.Tuple[float, float, float, float], height: int, width: int
) -> typing.Tuple[float, float, float, float]:
    x, y, w, h = coords
    return x * width, y * height, w * width, h * height


class VideoReader:
    def __init__(self, video_path: str, index: int):
        self.reader = cv2.VideoCapture(video_path)
        self.index = index
        self.frame_index = 0

    def read_frame(self, fidx: int) -> np.ndarray:
        if fidx >= self.frame_index:
            for _ in range(fidx - self.frame_index):
                self.reader.grab()
        else:
            self.reader.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        self.frame_index = fidx
        ret, frame = self.reader.retrieve()
        if ret:
            return frame
        else:
            raise ValueError("No frame available")