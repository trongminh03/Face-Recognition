import cv2 
import os 
import numpy as np 

dataset_path = 'dataset/train' 
# train_path = os.path.join(dataset_path, 'train') 

# Define Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml') 

# Define constants for image preprocessing
IMG_SIZE = 100
GRAYSCALE = True 

# Initialize empty lists to hold face images and labels
X_train = []
Y_train = []

# Loop over each person in the dataset
for person_name in os.listdir(dataset_path):
    # Define path to the person's image directory
    person_path = os.path.join(dataset_path, person_name)
    
    # Loop over each image in the person's directory
    for img_name in os.listdir(person_path):
        # Read the image
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)

        print(img_path)
        
        # Convert to grayscale if necessary
        if GRAYSCALE:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
        
        # Extract the face regions from the image
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            
            # Resize the face to a fixed size
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            
            # Store the flattened face array and corresponding label
            X_train.append(face.flatten())
            Y_train.append(person_name)
            
# Convert lists to numpy arrays
X_train = np.array(X_train)
Y_train = np.array(Y_train)

print("X_train", X_train) 
print("Y_train", Y_train)

# Save the training data as .npy files
np.save('X_train.npy', X_train)
np.save('Y_train.npy', Y_train)