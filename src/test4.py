import numpy as np 
import cv2 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml') 
IMG_SIZE = 100

# Load training data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy') 

# Train SVM classifier
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

cap = cv2.VideoCapture(0) 

while True: 
    # Capture frame-by-frame 
    ret, frame = cap.read()
    
    # Convert frame to grayscale 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    
    # Detect faces in grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5) 

    # For each detected face, do face recognition
    for (x, y, w, h) in faces:
        # Crop face region from grayscale image
        face_gray = gray[y:y+h, x:x+w]
        
        # Resize face region to fixed size
        face_gray_resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
        
        # Flatten resized face region to 1D array
        face_flat = face_gray_resized.flatten()
        
        # Predict label for face region using SVM classifier
        label = svm.predict([face_flat])[0]
        
        # Draw bounding box around face and label it with predicted name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)
    
    # Exit program when 'q' key is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release video capture object and close windows
cap.release()
cv2.destroyAllWindows()