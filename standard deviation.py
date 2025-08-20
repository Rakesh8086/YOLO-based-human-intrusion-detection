import math
import cv2
import numpy as np
import time


# Load YOLO
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
output_layers = net.getUnconnectedOutLayersNames()

conf_threshold = 0.5
conf_threshold_less = 0
#box_counter = 0
cap = cv2.VideoCapture("video1.mp4")

output_window_width = 800
output_window_height = 600

frame_counter = 0
nth_frame = 30
num_detections = 0
num_detections_less = 0
conf_sum = 0
conf_sum_less = 0
confidence_at_every_frame = []

total_frame_counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    # Process every nth frame
    if frame_counter % nth_frame == 0:
        height, width, channels = frame.shape
        total_frame_counter += 1
        #print("total_frame_counter = ", total_frame_counter)

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)


        class_ids = []
        confidences = []
        boxes = []

        # Process each output layer
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == 0 and confidence > conf_threshold:
                    conf_sum += confidence
                    num_detections += 1
                if class_id == 0 and confidence > conf_threshold_less:
                    conf_sum_less += confidence
                    num_detections_less += 1

                # Filter only 'person' class with a minimum confidence threshold
                #if class_id == 0 and confidence == 0:
                #    print('0')


                if class_id == 0 and confidence > conf_threshold:
                    # Rectangle coordinates
                    for i in range(len(boxes)):
                        if i in indices:
                            x, y, w, h = boxes[i]
                            label = ""
                            color = (0, 0, 255)  # Red
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)

                            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 4)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    #box_counter += 1
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    confidence_at_every_frame.append(float(confidence))
                    #print("Red", confidence)
                    class_ids.append(class_id)

                if class_id == 0 and confidence > conf_threshold_less and confidence < conf_threshold:
                    # Rectangle coordinates
                    for j in range(len(boxes)):
                        if j in indices_less:
                            x, y, w, h = boxes[j]
                            label = ""
                            color = (0, 255, 0)  # Green
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)

                            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 4)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    #box_counter += 1
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    confidence_at_every_frame.append(float(confidence))
                    #print("Green", confidence)
                    class_ids.append(class_id)
        # Apply non-max suppression to remove redundant overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.5)
        indices_less = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold_less, 0.5)

        # Draw bounding boxes and labels on the frame

        # Resize the frame to the desired output window size
        resized_frame = cv2.resize(frame, (output_window_width, output_window_height))

        # Display the output frame
        cv2.imshow("Human Detection", resized_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("e"):
            break

def standard_deviation():
    global confidence_at_every_frame
    global conf_sum
    conf_sum = sum(confidence_at_every_frame)
    len_confidence_at_every_frame = len(confidence_at_every_frame)
    mean_confidence = conf_sum / len_confidence_at_every_frame
    variance = 0
    for frame_confidence in confidence_at_every_frame:
        variance += (frame_confidence - mean_confidence) ** 2
    variance /= len_confidence_at_every_frame

    standard_deviation_of_confidence = math.sqrt(variance)
    #print(mean_confidence)
    return standard_deviation_of_confidence

#average_confidence = conf_sum / max(num_detections, 1)
# Release the video capture and close all windows

std_dev = standard_deviation()
print("Standard deviation = ",  std_dev)
cap.release()
cv2.destroyAllWindows()
