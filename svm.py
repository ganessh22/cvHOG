import cv2
import joblib
from sklearn.svm import LinearSVC


class SVM:
    def __init__(self, library: str, file_path: typing.Optional[str] = None, svm_object: typing.Optional[typing.Union[cv2.ml.SVM, LinearSVC]] = None):
        assert library in {"sklearn", "cv2"}
        assert library is not None or svm_object is not None
        self.library = library
        if file_path is None:
            self.svm = self.load_svm(file_path, library)
        else:
            self.svm = svm_object

    def load_svm(self, file_path: str, library: str):
        if self.library == "sklearn":
            return joblib.load(file_path)
        # TO-DO: loading cv2
        pass

    def save_svm(self, file_path: str):
        if self.library == "sklearn":
            joblib.dump(self.svm, file_path)
        else:
            joblib.dump(SVM.cv2_to_sklearn(self.svm), file_path)

    @staticmethod
    def sklearn_to_cv2(svm_object: LinearSVC) -> cv2.ml.SVM:
        pass

    @staticmethod
    def cv2_to_sklearn(svm_object: LinearSVC) -> cv2.ml.SVM:
        pass
    