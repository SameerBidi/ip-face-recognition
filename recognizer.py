import cv2
import numpy as np

# Initialize the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('path/to/haarcascade_frontalface_default.xml')

# Define the labels for each person
labels = ['person1', 'person2', 'person3']

# Load the images
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']

# Initialize the lists to store the cropped faces and corresponding labels
faces = []
face_labels = []

# Loop over the image paths
for i, image_path in enumerate(image_paths):
    
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
    # Loop over the faces and crop them
    for (x, y, w, h) in faces_rect:
        face = gray[y:y+h, x:x+w]
        faces.append(face)
        face_labels.append(i)
        
# Train the recognizer on the faces and labels
recognizer.train(faces, np.array(face_labels))

# Load the test image
test_img = cv2.imread('path/to/test_image.jpg')
gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

# Detect faces in the test image
test_faces_rect = face_cascade.detectMultiScale(gray_test, scaleFactor=1.2, minNeighbors=5)

# Loop over the faces in the test image
for (x, y, w, h) in test_faces_rect:
    
    # Crop the face
    test_face = gray_test[y:y+h, x:x+w]
    
    # Use the recognizer to predict the label for the test face
    label, confidence = recognizer.predict(test_face)
    
    # Draw a rectangle around the face and label it with the predicted name
    cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(test_img, labels[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
# Show the result
cv2.imshow('Result', test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
