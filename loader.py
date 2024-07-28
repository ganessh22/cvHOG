import glob
import os

from utils import get_file_by_ext_from_dir


class Data:
    def __init__(self, path: str, augment: bool = False):
        """
        Path should be of the format
        - path
            - vid1
                - vid1.mp4
                - vid1.json
            - vid2
            ....
        """
        self.path = path
        self.augment = augment
        self.setup()

    def setup(self):
        self.video_paths = []
        self.info = []
        for video_dir in glob.glob(os.path.join(self.path, "*/")):
            video_path = get_file_by_ext_from_dir(video_dir, "mp4")
            json_path = get_file_by_ext_from_dir(video_dir, "json")
            if video_path is None or json_path is None:
                print(f"Skipping {video_dir}...")
                continue
            self.video_paths.append(video_path)
            self.info.append(self.load_data(json_path))

    def load_data(self, json_file_path: str):
        pass

    def _load_iter(self, idx: int) -> typing.Dict:
        if self.augment:
            pass
            # TO-DO: train load
        else:
            pass
            # TO-DO: test load

    def __iter__(self):
        pass
