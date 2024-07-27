import glob
import os
import typing


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
