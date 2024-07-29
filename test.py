import argparse

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from classifier import HOGClassifier
from loader import Data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("test_dir", type=str)
    parser.add_argument("-m", "--model", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_data = Data(args.test_dir, augment=False)
    clf = HOGClassifier(args.model)
    ys, y_preds = [], []
    for (img, crops), y in iter(test_data):
        y_pred = clf.predict(img, crops, visualize=False)[0]
        ys.append(y)
        y_preds.append(y_pred)
    for fn, name in zip(
        [accuracy_score, f1_score, precision_score, recall_score],
        ["Accuracy", "F1", "Precision", "Recall"],
    ):
        print(f"{name} score : {fn(ys, y_preds)}")
