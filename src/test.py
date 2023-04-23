import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people

path = "dataset/"
# Load LFW dataset
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, h, w = lfw_people.images.shape
X = lfw_people.data
n_features = X.shape[1] 
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0] 

print("width: ", w, "height: ", h) 
print("n_features: ", n_features, "n_classes: ", n_classes) 

# Dataset Details
print("Number of Data Samples: % d" % n_samples)
print("Size of a data sample: % d" % n_features)
print("Number of Class Labels: % d" % n_classes)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(lfw_people.data, lfw_people.target, test_size=0.25, random_state=42)

# Use PCA to reduce dimensionality of the input data
n_components = 150
# pca = PCA(n_components=n_components, whiten=True)
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
pca.fit(X_train)

eigenfaces = pca.components_.reshape((n_components, h, w))
print("Projecting the input data on the eigenfaces orthonormal basis")
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# param_grid = {'svc__C': [1, 5, 10,50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
clf = GridSearchCV(
    SVC(kernel ='rbf', class_weight ='balanced'), param_grid
) 
clf = clf.fit(X_train_pca, y_train)
print("Best estimator found by grid search:")
print(clf.best_estimator_)

print("Predicting people's names on the test set")
y_pred = clf.predict(X_test_pca)

# # Train an SVM classifier on the reduced data
svc = SVC(kernel='rbf', class_weight='balanced')
clf = make_pipeline(pca, svc)
clf.fit(X_train, y_train)

# Test the classifier on the testing set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Use the classifier to recognize faces in an input image
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
# IMG_SIZE = 1850 
WIDTH = 50 
HEIGHT = 37     
# cap = cv2.VideoCapture(0) 
# img = cv2.imread('test.jpg')
# while True: 
#     # ret, frame = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
#     for (x, y, w, h) in faces:
#         face = gray[y:y+h, x:x+w]
#         face_resized = cv2.resize(face, (WIDTH, HEIGHT))
#         face_flat = face_resized.flatten()
#         # face_flat_pca = pca.transform(face_resized)
#         face_pred = clf.predict([face_flat])[0] 
#         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(img, str(lfw_people.target_names[face_pred]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#     cv2.imshow('Face Recognition', img)
#     # Exit program when 'q' key is pressed
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break
  
# cv2.waitKey(0)
# cv2.destroyAllWindows()