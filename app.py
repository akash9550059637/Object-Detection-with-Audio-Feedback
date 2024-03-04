import numpy as np
from flask import Flask, render_template, Response
import cv2
import threading
import time

from main import COLORS, CLASSES

app = Flask(__name__, template_folder='templates')

vs = cv2.VideoCapture(0)
frame_lock = threading.Lock()

# Initialize variables used for object detection
capturing = False
confidence_level = 0.6

# Load the pre-trained model and other necessary configurations
net = cv2.dnn.readNetFromCaffe('C:/Users/Akash/PycharmProjects/ssd/MobileNetSSD_deploy.prototxt',
                               'C:/Users/Akash/PycharmProjects/ssd/MobileNetSSD_deploy.caffemodel')

# Set preferable backend and target to CUDA if use_gpu is True
use_gpu = True
if use_gpu:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

def object_detection(frame):
    global capturing

    if capturing:
        # Resize the frame to the desired width for object detection
        frame = cv2.resize(frame, (400, 400))

        # Perform object detection
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_level:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([400, 400, 400, 400])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw bounding box and label on the frame
                label = f"{CLASSES[idx]}: {confidence:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS[idx], 1)

        # Display the frame with object detection
        cv2.imshow('Object Detection', frame)

def generate_frames():
    global vs
    global frame_lock

    while True:
        with frame_lock:
            ret, frame = vs.read()

        if not ret:
            break

        # Perform object detection on the frame
        object_detection(frame)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in the response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_object')
def capture_object():
    global capturing
    capturing = True

    # Simulate the 'a' key press for demonstration purposes
    # You should replace this with your actual logic from test.py
    # ...

    time.sleep(5)  # Simulate some processing time (replace with actual processing)

    # Simulate the 'r' key press for demonstration purposes
    # You should replace this with your actual logic from test.py
    # ...

    # Simulate the 'q' key press for demonstration purposes
    # You should replace this with your actual logic from test.py
    # ...

    capturing = False

    return "Object captured!"

if __name__ == '__main__':
    app.run(debug=True, port=8000)

