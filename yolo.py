import cv2
import numpy as np
from telegram import Bot, InputFile
import io
import asyncio
import time


frametime = []
# Load YOLO
def load_yolo(weights_file, cfg_file):
    net = cv2.dnn.readNet(weights_file, cfg_file)
    output_layers = net.getUnconnectedOutLayersNames()
    return net, output_layers

event_loop = asyncio.new_event_loop()
asyncio.set_event_loop(event_loop)
# Detect persons in a frame
# Detect persons in a frame and reduce noise in the region where the box is drawn
def detect_persons(frame, net, output_layers, conf_threshold=0, nms_threshold=0.5):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # class id 0 for detecting humans
            if class_id == 0 and confidence > conf_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Apply non-max suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = indices[0]
        x, y, w, h = boxes[i]
        roi = frame[y:y + h, x:x + w]  # Extract the region where the box is drawn
        roi = cv2.GaussianBlur(roi, (5, 5), 0)  # Apply Gaussian blur for noise reduction
        frame[y:y + h, x:x + w] = roi  # Replace the original region with noise-reduced region

    return indices, confidences, boxes


# Send alert message with the detected frame
async def send_alert_message(event_loop, bot, chat_id, frame):
    _, img_encoded = cv2.imencode('.png', frame)
    frame_bytes = img_encoded.tobytes()
    alert_message = 'Alert: Intruder detected in your surveillance camera!'

    while True:
        try:
            await bot.send_photo(chat_id=chat_id, photo=InputFile(io.BytesIO(frame_bytes)), caption=alert_message)
            break  # If successful, exit the loop
        except Exception as e:
            time.sleep(1) # Retry if unsuccessful
# Main function for processing video frames
def process_video(video_file, weights_file, cfg_file):
    cap = cv2.VideoCapture(video_file)
    net, output_layers = load_yolo(weights_file, cfg_file)
    frame_counter = 0
    nth_frame = 30
    box_counter = 0

    bot_token = '6388303298:AAFOr3RcWA9I4NhmbYiOuvs1FPSEu3P4Z9k'
    bot = Bot(token=bot_token)
    user_chat_id = '5968350801'  # Replace with the chat ID of the user

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            if frame_counter % nth_frame == 0:
                green_box = 0
                red_box = 0

                indices, confidences, boxes = detect_persons(frame, net, output_layers)

                for i in range(len(boxes)):
                    if i in indices:
                        x, y, w, h = boxes[i]

                        confidence = confidences[i]

                        if confidence > 0.5:
                            color = (0, 255, 0)  # Green color for confidence > 0.5
                        else:
                            color = (0, 255, 0)  # Red color for confidence <= 0.5

                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)
                        box_counter += 1

                        if box_counter % 10 == 0 and box_counter <= 100:
                            event_loop.run_until_complete(send_alert_message(event_loop, bot, user_chat_id, frame))
                        if box_counter % 50 == 0 and box_counter > 100:
                            event_loop.run_until_complete(send_alert_message(event_loop, bot, user_chat_id, frame))

                resized_frame = cv2.resize(frame, (800, 600))
                cv2.imshow("Human Detection", resized_frame)

            if cv2.waitKey(1) & 0xFF == ord("e"):
                break

    except KeyboardInterrupt:
        pass

    finally:
        cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    video_file = "video.mp4"
    weights_file = "yolov4.weights"
    cfg_file = "yolov4.cfg"
    process_video(video_file, weights_file, cfg_file)