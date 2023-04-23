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

dataset_path = 'dataset/train' 
WIDTH =  64
HEIGHT = 64
image_size = (WIDTH, HEIGHT) 

images = [] 
labels = []

for person_name in os.listdir(dataset_path): 
    person_path = os.path.join(dataset_path, person_name) 
    for img_name in os.listdir(person_path): 
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)  
        # height, width = img.shape[:2] 
        # new_width = int(width * 0.4) 
        # new_height = int(height * 0.4)
        # image = cv2.resize(img, (new_width, new_height))
        image = cv2.resize(img, image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        # images.append(image.flatten())
        labels.append(person_name)

X = np.array(images)   
y = np.array(labels) 
print(X.shape)
# print("sample", "features", X.shape[0], X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# n_components = 30
# # pca = PCA(n_components=n_components, whiten=True)
# pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
# pca.fit(X_train)

# # eigenfaces = pca.components_.reshape((n_components, 64, 64))
# print("Projecting the input data on the eigenfaces orthonormal basis")
# X_train_pca = pca.transform(X_train)
# X_test_pca = pca.transform(X_test) 

# param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
#               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], } 
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma' : [0.0001, 0.001, 0.1,1],
    'kernel' : ['rbf', 'poly']
}
# svc = svm.SVC(probability=True)
print("Starting training, please wait ...")
# param_grid = {'svc__C': [1, 5, 10,50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
clf = GridSearchCV(
    # SVC(kernel ='rbf', class_weight ='balanced'), param_grid
    SVC(probability=True), param_grid
) 
# clf = clf.fit(X_train_pca, y_train) 
clf = clf.fit(X_train, y_train) 

print("Best estimator found by grid search:")
print(clf.best_estimator_)

print("Predicting people's names on the test set")
# y_pred = clf.predict(X_test_pca)
y_pred = clf.predict(X_test)

# # Train an SVM classifier on the reduced data
svc = SVC(kernel='rbf', class_weight='balanced')
# clf = make_pipeline(pca, svc)
clf.fit(X_train, y_train)

# Test the classifier on the testing set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy) 

# Use the classifier to recognize faces in an input image
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
# IMG_SIZE = 1850 
# WIDTH = 64 
# HEIGHT = 64
# cap = cv2.VideoCapture(0) 
img = cv2.imread('test2.jpg')
while True: 
    # ret, frame = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (WIDTH, HEIGHT))
        face_flat = face_resized.flatten()
        # face_flat_pca = pca.transform(face_resized)
        face_pred = clf.predict([face_flat])[0]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, str(face_pred), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Face Recognition', img)
    # Exit program when 'q' key is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
  
cv2.waitKey(0)
cv2.destroyAllWindows()