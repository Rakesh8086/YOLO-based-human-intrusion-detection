import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import load

# Load the trained SVM model
model_filename = "svm_model.joblib"
svm_model = load(model_filename)

# Create a HOG descriptor
win_size = (64, 128)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

# Function to perform human detection in a video
def detect_humans_in_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Resize the frame if needed
        # frame = cv2.resize(frame, (width, height))

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform HOG feature extraction
        hog_features = hog.compute(gray)

        # Reshape the feature vector to be compatible with SVM prediction
        hog_features = hog_features.reshape(1, -1)

        # Predict using the trained SVM model
        prediction = svm_model.predict(hog_features)

        # Display the result
        if prediction == 1:  # 1 indicates the presence of a human
            cv2.putText(frame, "Human Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Human", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Human Detection", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Specify the path to the video file
video_path = "video1.mp4"

# Perform human detection in the video
detect_humans_in_video(video_path)
