# import useful libraries
import cv2
import numpy as np
import subprocess
import os
import sys
from yolo_utils import *

# Detect platform: use Picamera2 on Raspberry Pi, webcam otherwise
USE_PICAM = False
try:
    from picamera2 import Picamera2
    USE_PICAM = True
except ImportError:
    pass

# video file names
temp_video = "temp_recording.avi"
output_video = "recording.mp4"

# check OpenCV
print("OpenCV version :", cv2.__version__)
cuda_count = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, 'cuda') else 0
print("Available CUDA devices:", cuda_count, "\n")

# load class names
obj_file = './obj.names'
classNames = read_classes(obj_file)
print("Classes' names :", classNames, "\n")

# load YOLO model
modelConfig_path = './cfg/yolov4.cfg'
modelWeights_path = './weights/yolov4.weights'

neural_net = cv2.dnn.readNetFromDarknet(modelConfig_path, modelWeights_path)
if cuda_count > 0:
    neural_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    neural_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
    neural_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    neural_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

confidenceThreshold = 0.5
nmsThreshold = 0.1

network = neural_net
height, width = 128, 128   # input size for network

# initialize camera
if USE_PICAM:
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={'size': (640, 480)}))
    picam2.start()
else:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# setup Video Writer (AVI first)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(temp_video, fourcc, 30.0, (640, 480))

print("[MAIN] Recording started... Press Ctrl+C to stop.")

try:
    while True:
        if USE_PICAM:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from webcam.")
                break

        # object detection
        outputs = convert_to_blob(frame, network, height, width)
        bounding_boxes, class_objects, confidence_probs = object_detection(
            outputs, frame, confidenceThreshold)

        for i in range(len(bounding_boxes)):
            print(f"[Debug] Detected: Class={class_objects[i]}, Confidence={confidence_probs[i]:.2f}")
            # TODO: change the class number to the class number of traffic light in obj.names file
            if class_objects[i] == 3:
                # TODO: detect the color of the traffic light (red) by merging task 1
                # step 1: crop the bounding box area from the frame
                x, y, w, h = [int(v) for v in bounding_boxes[i]]
                cropped = frame[y:y+h, x:x+w]

                # step 2: convert the cropped area to HSV color space
                hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

                # step 3: create a mask for red color
                mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
                mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
                mask = mask1 | mask2

                # step 4: check if there are enough contour areas in the mask to confirm the traffic light is red
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 500:
                        # step 5: print a message if the traffic light is red
                        print("Red light detected!")
                        break
        

        indices = nms_bbox(
            bounding_boxes,
            confidence_probs,
            confidenceThreshold,
            nmsThreshold
        )

        box_drawing(
            frame,
            indices,
            bounding_boxes,
            class_objects,
            confidence_probs,
            classNames,
            color=(0, 255, 255),
            thickness=2
        )

        # write frame to video file
        out.write(frame)

except KeyboardInterrupt:
    print("\n[MAIN] Stopping recording...")

# cleanup
out.release()
if USE_PICAM:
    picam2.close()
else:
    cap.release()

print("[MAIN] Converting to MP4 using ffmpeg...")

subprocess.run(["ffmpeg", "-y", "-i", temp_video, "-vcodec", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p", output_video], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os.remove(temp_video)

print("[MAIN] Video saved successfully as", output_video)
