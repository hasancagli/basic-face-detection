import cv2 # opencv
import mediapipe as mp

capture = cv2.VideoCapture(0)

faceDetection = mp.solutions.face_detection.FaceDetection()
mpDraw = mp.solutions.drawing_utils

while True:
    success, image = capture.read()

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imageRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(image, detection)
            print(id, detection)

    cv2.imshow("Image", image)
    cv2.waitKey(1)
