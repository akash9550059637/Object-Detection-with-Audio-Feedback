from imutils.video import FPS
import numpy as np
import imutils
import cv2
import math

use_gpu = True
live_video = False
confidence_level = 0.6
temp = 0
tec = ""
fps = FPS().start()
ret = True

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor", "gun", "revolver"]

COLORS = np.random.uniform(0, 1, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe('C:/Users/Akash/PycharmProjects/ssd/MobileNetSSD_deploy.prototxt',
                               'C:/Users/Akash/PycharmProjects/ssd/MobileNetSSD_deploy.caffemodel')

if use_gpu:
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

print("[INFO] accessing video stream...")
if live_video:
    vs = cv2.VideoCapture(0)
else:
    vs = cv2.VideoCapture('test.mp4')

while ret:
    ret, frame = vs.read()
    if ret:
        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_level:
                idx = int(detections[0, 0, i, 1])
                temp += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = f"{CLASSES[idx]}: {confidence:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLORS[idx], 1)

        text = f"No of {tec} = {math.ceil(temp / 2)}"
        temp = 0

        frame = imutils.resize(frame, height=400)
        cv2.imshow('Live detection', frame)

        y = 15
        for i in text:
            if i == "\n":
                y += 15

        image = np.zeros((200, 200, 3), dtype=np.uint8)  # Create a black image for text display
        cv2.putText(image, text, (5, y), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255), 1)
        cv2.imshow("Text", image)

        if cv2.waitKey(1) == 27:
            break

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

vs.release()
cv2.destroyAllWindows()