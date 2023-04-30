import mediapipe as mp
import cv2 as cv

facemesh=mp.solutions.face_mesh
mpdraw=mp.solutions.drawing_utils
face=facemesh.FaceMesh(static_image_mode=False,min_detection_confidence=0.5,min_tracking_confidence=0.5)


cap=cv.VideoCapture(0)

while True:
    success,img=cap.read()
    imgrgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    result = face.process(imgrgb)

    if result.multi_face_landmarks:
        for i in result.multi_face_landmarks:
            mpdraw.draw_landmarks(img,i,facemesh.FACEMESH_CONTOURS,landmark_drawing_spec=mpdraw.DrawingSpec(color=(0,255,255),circle_radius=1))

    cv.imshow('image', img)
    cv.waitKey(1)