import cv2
import numpy as np

# Load SSD with Inception V2 model
net = cv2.dnn.readNetFromCaffe('mobilenet_ssd/MobileNetSSD_deploy.prototxt',
                               'mobilenet_ssd/MobileNetSSD_deploy.caffemodel')

# OpenCV's VideoCapture to read the video
cap = cv2.VideoCapture('video.mp4')
nth_frame_counter = 0
box_counter = 0
false_positives = 0
false_negatives = 0  # Initialize false negatives counter

# Create a list to store confidence scores for each frame
confidences = []

# Set parameters
input_width, input_height = 300, 300
confidence_threshold = 0.5
skip_frames = 30  # Number of frames to skip

# Resize the output window
output_width, output_height = 800, 640
cv2.namedWindow('SSD with Inception V2', cv2.WINDOW_NORMAL)
cv2.resizeWindow('SSD with Inception V2', output_width, output_height)

frame_counter = 0

while True:
    ret, frame = cap.read()

    # Break the loop if no more frames
    if not ret:
        break

    frame_counter += 1

    # Skip frames based on the skip_frames value
    if frame_counter % skip_frames != 0:
        continue

    nth_frame_counter += 1  # Increment counter for every processed frame

    # Resize frame to match model input size
    frame = cv2.resize(frame, (input_width, input_height))

    # Prepare input blob
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (input_width, input_height), 127.5)

    # Set the blob as input to the network
    net.setInput(blob)

    # Forward pass to get the detections
    detections = net.forward()

    # List to store confidence scores for the current frame
    frame_confidences = []

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Check if confidence is above a certain threshold
        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])

            # Bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([input_width, input_height, input_width, input_height])
            (x, y, w, h) = box.astype(int)

            # Draw the bounding box on the frame
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (w, h), color, 2)
            box_counter += 1

            # Display the class label and confidence
            label = f'Class: {class_id}, Confidence: {confidence:.2f}'
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Append confidence score to the list for the current frame
            frame_confidences.append(confidence)

    # Append the list of confidence scores to the main list
    confidences.append(frame_confidences)

    # Increase false negatives count if no human detected in the frame
    if not frame_confidences:
        false_negatives += 1

    # Display the resulting frame
    cv2.imshow('SSD with Inception V2', frame)

    # Break the loop if 'e' is pressed
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

# Calculate accuracy metrics
print("True positive: ", box_counter)
print("False negatives: ", false_negatives)
true_negatives = nth_frame_counter - (box_counter + false_negatives + false_positives)
print("True negatives: ", true_negatives)
print("False positives: ", false_positives)
precision = box_counter / (box_counter + false_positives) if (box_counter + false_positives) > 0 else 0
recall = box_counter / (box_counter + false_negatives) if (box_counter + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

accuracy_percentage = min(f1_score * 100, 100)
print("Accuracy: ", accuracy_percentage)

# Release the VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()
