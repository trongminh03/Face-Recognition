import cv2
import os
import math
from imutils import face_utils, rotate_bound
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


def calculate_inclination(point1, point2):
    x1, x2, y1, y2 = point1[0], point2[0], point1[1], point2[1]
    incl = 180 / math.pi * math.atan((float(y2 - y1)) / (x2 - x1))
    return incl

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml') 
eye_cascade = cv2.CascadeClassifier('cascades/eye.xml')

WIDTH = 100
HEIGHT = 100
check = True
cap = cv2.VideoCapture(0)

while True: 
    isSuccess, img = cap.read()
    img = np.pad(img, ((900, 900), (900, 900), (0, 0)), mode='constant', constant_values=0)
    # img = cv2.imread('test.jpg')
    # ret, frame = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        # roi_color = img[y:y + h, x:x + w] 
        roi_color = img[y:y + w, x:x + w] 
        face_resized = cv2.resize(face, (WIDTH, HEIGHT))
        face_flat = face_resized.flatten()
        # face_flat_pca = pca.transform(face_flat)
        # face_pred = clf.predict([face_flat])[0]
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.putText(img, str(face_pred), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        eyes = eye_cascade.detectMultiScale(face)  
        for (ex, ey, ew, eh) in eyes:
            if check == True:
                print(ex, ey, ew, eh)
            # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        glass = cv2.imread('images/sunglasses_2.png', cv2.IMREAD_UNCHANGED)
        # print(glass.shape)
        # transparent = glass[:,:,3] != 0
        # print(transparent.shape)

        if len(eyes) == 2 and abs(eyes[1][1] - eyes[0][1]) < 200:
            # for (ex, ey, ew, eh) in eyes:
            #     cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0,255,0), 2)
            if eyes[0][0] > eyes[1][0]:
                tmp = eyes[0][0]
                eyes[0][0] = eyes[1][0]
                eyes[1][0] = tmp
            s = (eyes[0][0], eyes[0][1])
            e = (eyes[1][0] + eyes[1][2], eyes[1][1])
            incl =  calculate_inclination(s, e)
            # if eyes[0][1] < eyes[1][1]:
            #     incl = calculate_inclination(s, e)
            glass = rotate_bound(glass, incl)
            glass_width = e[0] - s[0]
            glass_height = eyes[0][3]
            glass = cv2.resize(glass, (glass_width, glass_height))
            transparent_region = glass[:,:,3] != 0
            roi_color[s[1] : s[1] + glass_height, s[0] : s[0] + glass_width,:][transparent_region] = glass[:,:,:3][transparent_region]

        beard = cv2.imread('images/santa_filter.png', cv2.IMREAD_UNCHANGED)
        # beard = rotate_bound(beard, incl)
        beard_width = int(w)
        beard_height = int(beard_width*1.4)
        beard = cv2.resize(beard, (beard_width, beard_height))
        transparent_region = beard[:,:,3] != 0
        s = int(1.8*h/3)
        img[y + s : y + s + beard_height, x : x + beard_width, :][transparent_region] = beard[:,:,:3][transparent_region]

    
    check = False
    cv2.imshow('Face Recognition', img[900:-900, 900:-900, :])
    # Exit program when 'q' key is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
  
cv2.waitKey(0)
cv2.destroyAllWindows()