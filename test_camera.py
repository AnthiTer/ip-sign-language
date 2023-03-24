import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import time

mp_hol = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

# as pictures/videos from cv2 are in BGR form, we have to change it so they can be readable
def colour_conv(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    result = model.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, result

# showing the landmarks in the pictures/videos
def landmark_show(img, res):
    mp_draw.draw_landmarks(img, res.right_hand_landmarks, mp_hol.HAND_CONNECTIONS, 
    mp_draw.DrawingSpec(color = (0,0,256), thickness = 1, circle_radius = 2),
    mp_draw.DrawingSpec(color = (256,256,256), thickness = 1, circle_radius = 1))
    mp_draw.draw_landmarks(img, res.left_hand_landmarks, mp_hol.HAND_CONNECTIONS,
    mp_draw.DrawingSpec(color = (0,0,256), thickness = 1, circle_radius = 2),
    mp_draw.DrawingSpec(color = (256,256,256), thickness = 1, circle_radius = 1))
    # mp_draw.draw_landmarks(img, res.face_landmarks, mp_hol.FACEMESH_TESSELATION,
    # mp_draw.DrawingSpec(color = (0,0,256), thickness = 1, circle_radius = 2),
    # mp_draw.DrawingSpec(color = (256,256,256), thickness = 1, circle_radius = 1))
    mp_draw.draw_landmarks(img, res.pose_landmarks, mp_hol.POSE_CONNECTIONS,
    mp_draw.DrawingSpec(color = (0,0,256), thickness = 1, circle_radius = 2),
    mp_draw.DrawingSpec(color = (256,256,256), thickness = 1, circle_radius = 1))
    
# test camera to see if everything works
cap = cv2.VideoCapture(0)
with mp_hol.Holistic(min_detection_confidence=0.5, min_tracking_confidence = 0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        img, result = colour_conv(frame, holistic)
        landmark_show(img, result)
        cv2.imshow('Sign Language Recognition', img)
        cv2.namedWindow('Sign Language Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Sign Language Recognition', 800, 600)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

