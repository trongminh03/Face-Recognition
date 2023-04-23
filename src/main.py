import cv2
import os
from time import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people

dataset_path = 'dataset/val' 
WIDTH =  100
HEIGHT = 100
image_size = (WIDTH, HEIGHT) 

images = [] 
labels = []

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml') 
eye_cascade = cv2.CascadeClassifier('cascades/eye.xml')

# Load dataset to arrays
for person_name in os.listdir(dataset_path): 
    person_path = os.path.join(dataset_path, person_name) 
    for img_name in os.listdir(person_path): 
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)  
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect face
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, image_size)
            images.append(face_resized.flatten())  
            # images.append(face_resized)
            labels.append(person_name)

X = np.array(images)   
y = np.array(labels) 
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
print(len(X_train), len(X_test), len(y_train), len(y_test))

n_components = 200
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
pca.fit(X_train) 
# pca.fit_transform(X_train)

# eigenfaces = pca.components_.reshape((n_components, 64, 64))
# print("Projecting the input data on the eigenfaces orthonormal basis")
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], } 
print("Starting training, please wait ...")

clf = GridSearchCV(
    SVC(kernel ='rbf', class_weight ='balanced'), param_grid
) 
clf = clf.fit(X_train_pca, y_train) 

print("Best estimator found by grid search:")
print(clf.best_estimator_)

print("Predicting people's names on the test set")

y_pred = clf.predict(X_test_pca)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy) 
print(classification_report(y_test, y_pred))

# print("Confusion Matrix is:")
# print(confusion_matrix(y_test, y_pred, labels = range(7)))

# Use the classifier to recognize faces in an input image
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')

WIDTH = 20
HEIGHT = 10
# cap = cv2.VideoCapture(0) 
while True: 
    img = cv2.imread('test.jpg')
    # ret, frame = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        roi_color = img[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (WIDTH, HEIGHT))
        face_flat = face_resized.flatten()
        # face_flat_pca = pca.transform(face_flat)
        face_pred = clf.predict([face_flat])[0]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, str(face_pred), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        eyes = eye_cascade.detectMultiScale(face)  
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2.imshow('Face Recognition', img)
    # Exit program when 'q' key is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
  
cv2.waitKey(0)
cv2.destroyAllWindows()