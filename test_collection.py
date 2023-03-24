import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import time

mp_hol = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
DATA_PATH_HAND = os.path.join('/Users/anthi/Documents/IP/test_signs_hands')
actions = np.array(['whatever'])
no_vid = 6
frames = 30
start = 0

def rec(x):
    return 'b_0'+ str(x)

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

def extract_all_keypoints(results):
    lhand = np.array([[res.x, res.y, res.z] 
        for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rhand = np.array([[res.x, res.y, res.z] 
        for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    face = np.array([[res.x, res.y, res.z] 
        for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
        for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return np.concatenate([lhand, rhand, face, pose])

def extract_hand_keypoints(results):
    lhand = np.array([[res.x, res.y, res.z] 
        for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rhand = np.array([[res.x, res.y, res.z] 
        for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lhand, rhand])

def extract_pose_hand_keypoints(results):
    lhand = np.array([[res.x, res.y, res.z] 
        for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rhand = np.array([[res.x, res.y, res.z] 
        for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
        for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return np.concatenate([lhand, rhand, pose])

for x in actions:
    for y in range(no_vid+start):
        try: 
            os.makedirs(os.path.join(DATA_PATH_HAND, x, rec(y)))
        except:
            pass

cap = cv2.VideoCapture(0)
with mp_hol.Holistic(min_detection_confidence=0.5, min_tracking_confidence = 0.5) as holistic:
    for x in actions:
        for y in range(start, start + no_vid):
            for f in range(frames):
                ret, frame = cap.read()
                img, result = colour_conv(frame, holistic)
                landmark_show(img, result)

                if f == 0:
                    cv2.putText(img, 'STARTING COLLECTION', (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0, 0), 4, cv2.LINE_AA)
                    cv2.putText(img, 'COLLECTING FRAMES FOR {} VIDEO NO.{}'.format(x,y), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                    cv2.imshow('Sign Language Recognition', img)
                    cv2.namedWindow('Sign Language Recognition', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Sign Language Recognition', 800, 600)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(img, 'COLLECTING FRAMES FOR {} VIDEO NO.{}'.format(x,y), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,0,255), 2, cv2.LINE_AA)
                    cv2.imshow('Sign Language Recognition', img)
                    cv2.namedWindow('Sign Language Recognition', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Sign Language Recognition', 800, 600)

                keypoints_all = extract_all_keypoints(result)
                keypoints_hand = extract_hand_keypoints(result)
                keypoints_pose = extract_pose_hand_keypoints(result)

                path_npy_hand = os.path.join(DATA_PATH_HAND, x, rec(y), str(f))
               
                np.save(path_npy_hand, keypoints_hand)
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    cap.release()
    cv2.destroyAllWindows()