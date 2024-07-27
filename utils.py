import glob
import os
import typing


def get_file_by_ext_from_dir(directory: str, ext: str) -> typing.Optional[str]:
    files = list(glob.glob(os.path.join(directory, f"*.{ext}")))
    if len(files) == 0:
        return None
    return files[0]
