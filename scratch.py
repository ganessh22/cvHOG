# import cv2
# import joblib  # For loading the trained sklearn model
# import numpy as np
# from sklearn.svm import LinearSVC

# # Load your trained LinearSVC model
# # Make sure to replace 'your_model.pkl' with the path to your trained model
# model = joblib.load("your_model.pkl")

# # Extract the weights and bias from the LinearSVC model
# weights = model.coef_[0]  # This is a 1D array of shape (3780,)
# bias = model.intercept_[0]  # This is a scalar

# # Create an OpenCV SVM model
# svm = cv2.ml.SVM_create()

# # Set the SVM type and kernel
# svm.setKernel(cv2.ml.SVM_LINEAR)
# svm.setType(cv2.ml.SVM_C_SVC)
# svm.setC(1.0)  # Set the regularization parameter, adjust as needed

# # Set the weights and bias for the OpenCV SVM model
# # OpenCV expects weights in the form of a 2D array (1, 3780)
# svm.setSupportVectors(np.array([weights], dtype=np.float32))
# svm.setDecisionFunction(np.array([bias], dtype=np.float32))

# # Save the OpenCV SVM model if needed
# svm.save("converted_model.xml")

# print("Model successfully converted and saved as 'converted_model.xml'")


# import cv2
# import joblib
# import numpy as np
# from sklearn.svm import LinearSVC

# # Load your trained LinearSVC model
# model_sklearn = joblib.load("your_model.pkl")

# # Load the converted OpenCV SVM model
# svm_opencv = cv2.ml.SVM_load("converted_model.xml")

# # Generate a random 3780-dimensional vector
# random_vector = np.random.rand(1, 3780).astype(np.float32)

# # Predict using sklearn's LinearSVC
# prediction_sklearn = model_sklearn.predict(random_vector)

# # Predict using OpenCV's SVM
# # OpenCV's predict function returns a tuple, we need the first element for the actual prediction
# _, prediction_opencv = svm_opencv.predict(random_vector)

# # Convert the prediction from OpenCV to a simple integer (it returns a float array)
# prediction_opencv = prediction_opencv[0][0]

# # Print the predictions for verification
# print(f"Prediction from sklearn LinearSVC: {prediction_sklearn[0]}")
# print(f"Prediction from OpenCV SVM: {int(prediction_opencv)}")

# # Assert that both predictions are equal
# assert prediction_sklearn[0] == int(
#     prediction_opencv
# ), "Predictions do not match!"

# print("Both predictions are equal!")


import cv2
import numpy as np

# Set the random seed for reproducibility
np.random.seed(0)

# Generate random data with 3780 features and 100 samples
data = np.random.rand(100, 3780).astype(np.float32)
labels = np.random.randint(0, 2, size=(100, 1)).astype(np.int32)  # Binary labels

# Create and train the SVM
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10000, 1e-6))
svm.train(data, cv2.ml.ROW_SAMPLE, labels)

# Save the trained model to an XML file
svm.save('svm_model.xml')