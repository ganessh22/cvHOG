import typing

import cv2
import joblib
import numpy as np
from sklearn.svm import LinearSVC


class SVM:
    def __init__(
        self,
        library: str,
        file_path: typing.Optional[str] = None,
        svm_object: typing.Optional[
            typing.Union[cv2.ml.SVM, LinearSVC]
        ] = None,
    ):
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
        weights = svm_object.coef_[0]
        bias = svm_object.intercept_[0]
        svm = cv2.ml.SVM_create()
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setC(1.0)  # Set the regularization parameter, adjust as needed
        # Set the weights and bias for the OpenCV SVM model
        # OpenCV expects weights in the form of a 2D array (1, 3780)
        svm.setSupportVectors(np.array([weights], dtype=np.float32))
        svm.setDecisionFunction(np.array([bias], dtype=np.float32))
        return svm

    @staticmethod
    def cv2_to_sklearn(svm_object: cv2.ml.SVM) -> LinearSVC:
        # Extract the weights and bias from the OpenCV SVM model
        weights = svm_object.getSupportVectors()[
            0
        ].T  # Transpose to match LinearSVC format
        bias = (
            -1 * svm_object.getDecisionFunction(0)[0]
        )  # Negate the bias term
        # Create a LinearSVC model in scikit-learn
        svm_sklearn = LinearSVC(random_state=42)
        # Set the weights and bias for the LinearSVC model
        svm_sklearn.coef_ = weights.reshape(
            1, -1
        )  # Reshape to match LinearSVC format
        svm_sklearn.intercept_ = np.array([bias])
        return svm_sklearn


def get_default_cv2_hog_svm() -> cv2.ml.SVM:
    support_vectors = cv2.HOGDescriptor_getDefaultPeopleDetector()
    # The support vectors are already in the right format for OpenCV SVM
    # The bias term is the last element of the support vectors
    bias = support_vectors[-1]  # Last element is the bias
    weights = support_vectors[:-1]  # All but the last element are the weights
    # Step 3: Create an OpenCV SVM model
    svm = cv2.ml.SVM_create()
    # Set the SVM type and kernel
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(1.0)  # Set the regularization parameter, adjust as needed
    # Step 4: Set the support vectors and the decision function
    # OpenCV expects weights in the form of a 2D array (1, n_features)
    svm.setSupportVectors(np.array([weights], dtype=np.float32))
    svm.setDecisionFunction(np.array([bias], dtype=np.float32))
    return svm


def test_cv2_and_sklearn():
    cv2_svm = get_default_cv2_hog_svm()
    random_vector = np.random.rand(1, 3780).astype(np.float32)
    sklearn_svm = SVM.cv2_to_sklearn(cv2_svm)
    prediction_sklearn = sklearn_svm.predict(random_vector)
    prediction_cv2 = cv2_svm.predict(random_vector)[1]
    np.testing.assert_array_almost_equal(prediction_sklearn, prediction_cv2)


if __name__ == "__main__":
    test_cv2_and_sklearn()
