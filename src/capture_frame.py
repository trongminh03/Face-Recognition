import cv2
import os
import time

# Create a directory to store the images
folder_name = "Le Trong Minh"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Set up the camera
camera = cv2.VideoCapture(0)  # Use 0 if you have only one camera connected

# Capture 100 images
count = 0
while count < 100:
    # Capture frame-by-frame
    ret, frame = camera.read()

    # Display the resulting frame
    cv2.imshow("Camera", frame)

    # Save the image
    image_name = f"{folder_name}/image_{count}.jpg"
    cv2.imwrite(image_name, frame)

    # Increment the count
    count += 1

    # Wait for 1 second
    time.sleep(1)

    # Check if the user wants to stop capturing images
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()
