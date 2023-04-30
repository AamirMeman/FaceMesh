import cv2 as cv
import mediapipe as mp

cap=cv.VideoCapture(0)

mphands=mp.solutions.hands
hands=mphands.Hands()
mpdraw=mp.solutions.drawing_utils

while True:
    success,img=cap.read()
    imgrgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    result=hands.process(imgrgb)

    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            mpdraw.draw_landmarks(img,handlms,mphands.HAND_CONNECTIONS)

    cv.imshow("image",img)
    cv.waitKey(1)