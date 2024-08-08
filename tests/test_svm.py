import cv2
import numpy as np
from sklearn.svm import LinearSVC

from ..svm import SVM, get_default_cv2_hog_svm


def test_cv2_and_sklear():
    cv2_svm = get_default_cv2_hog_svm()
    random_vector = np.random.rand(1, 3780).astype(np.float32)
    sklearn_svm = SVM.cv2_to_sklearn(cv2_svm)
    prediction_sklearn = sklearn_svm.predict(random_vector)
    prediction_cv2 = cv2_svm.predict(random_vector)[1]
    np.testing.assert_array_almost_equal(prediction_sklearn, prediction_cv2)
