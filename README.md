# cvHOG
## TO-DO
* The test script needs some work. The crops info is not sufficient for testing. It is needed for train. For test we need the rough crop around the motion contour enclosing box. 
* For train we need a script to make a numpy array for training.

### Note
So the cv2 SVM's parameters can be gotten using below as shown in the [link](https://stackoverflow.com/questions/76667072/how-to-use-custom-svm-detector-with-cv2-hogdescriptor)
```python
model = cv2.ml.SVM_load("model.svm")
support_vectors = model.getSupportVectors()
coefficients = -model.getDecisionFunction(0)[0]

coefficients = np.array(coefficients).reshape(1, -1)
svmdetector = np.concatenate((support_vectors, coefficients), axis=1)
hog.setSVMDetector(svmdetector.T.flatten())
```
this gives a vector which is n_features + 1. the default people SVM in cv2 also is 3781. The one trained with cv2 + sklearn is also 3780 input. I think we can convert one to the other using the two attributes. Need to test.
```
coef_ndarray of shape (1, n_features) if n_classes == 2 else (n_classes, n_features)
Weights assigned to the features (coefficients in the primal problem).

coef_ is a readonly property derived from raw_coef_ that follows the internal memory layout of liblinear.

intercept_ndarray of shape (1,) if n_classes == 2 else (n_classes,)
Constants in decision function.
```
Current status is I can convert an cv2 SVM to sklearn but not the other way. Probably I will need to modify the XML file. Not sure if I want to do that as finally I just need the vector which I can get from sklearn and cv2 too. 

I think this repo is getting too big. I need to simplify. 
Only 3 components are needed here : 
* take two folders of images and train and test an SVM with HOG
* convert SVM to cv2 format i.e. array
* infer on an image function

Something wrong with sklearn hog features, doesn't match sklearn even when params are all same. But could be wrong order. Need to do a pairwise subtraction and plot the matrix.