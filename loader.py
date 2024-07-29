import glob
import json
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
            self.index_map = self._make_index_map()
        self.current = 0

    def _make_index_map(self) -> typing.List[typing.Tuple[int, int]]:
        index_map = []
        for i, info_idx in enumerate(self.info):
            len_crops = len(info_idx["crops"])
            if len_crops == 0:
                index_map.append((i, 0))
                continue
            for j in range(len_crops):
                index_map.append((i, j))
        return index_map

    def load_data(self, json_file_path: str):
        """data format : 
        frames : [1, 5, 10, ...],
        crops: [[], [(0.0, 0.0, 1.0, 0.9), ...], [], ...],
        gt: [[False], [True, True]] ... ],
        len(frames) == len(crops) == len(gt)
        each element of crops is a list of tuples(float, float, float, float) i.e. a crop
        """
        with open(json_file_path, "r") as f:
            return json.load(f)

    def _load_pure_data(self, idx: int) -> typing.Dict:
        pass

    def _augment_data(self, data: typing.Dict) -> typing.Dict:
        # TO-DO: update the typing.Dicts in this file to be proper
        pass

    def _load_iter(self, idx: int) -> typing.Dict:
        data = self._load_pure_data(idx)
        if self.augment:
            data = self._augment_data(data)
        return data

    def __iter__(self):
        return self

    def __next__(self):
        if self.current == len(self):
            raise StopIteration
        self.current += 1
        return self._load_iter(self.current - 1)

    def __len__(self) -> int:
        return len(self.index_map)
