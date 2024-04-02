import cv2
import os

# Initialize camera
cap = cv2.VideoCapture(0)

# Load face cascade
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")

# Counter for capturing images
count = 0

# Input user's name
name = input("Enter your name: ")

# Create a directory for the user's dataset
user_dataset_dir = os.path.join("images", name)
if not os.path.exists(user_dataset_dir):
    os.makedirs(user_dataset_dir)

while True:
    ret, frame = cap.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

        # Crop the face region
        face_roi = frame[y:y+h, x:x+w]

        # Resize the face to 100x100 pixels
        resized_face = cv2.resize(face_roi, (100, 100))

        # Save the resized face (in color)
        cv2.imwrite(f"{user_dataset_dir}/{count}.jpg", resized_face)

        # Display the image
        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 300:  # Capture 300 images or press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
