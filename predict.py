import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.layers import LSTM, Dense, LeakyReLU
from tensorflow.keras.models import Sequential
from islr_network import islr

sequence = []
sentence = []
predictions = []
threshold = 0.7

mp_hol = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

def colour_conv(img, model):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    result = model.process(img)
    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, result

def normalize_data(X):
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / X_std
        return X_norm
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

def extract_pose_hand_keypoints(results):
    lhand = np.array([[res.x, res.y, res.z] 
        for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rhand = np.array([[res.x, res.y, res.z] 
        for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
        for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return np.concatenate([lhand, rhand, pose])

# actions = np.array(['goodbye', 'hi', 'you', 'me', 'thankyou', 'goodmorning'])

actions = np.array(['Come', 'Good', 'Happy', 'Home', 'I Love You', 'Sorry'])

# actions = np.array(['come', 'good', 'goodbye', 'goodmorning', 'happy', 'hi', 'home', 'iloveyou', 'me', 'sorry', 'thankyou', 'you'])

# actions = np.array(['come', 'good', 'goodbye', 'goodmorning', 'happy', 'hi', 'home', 'iloveyou', 'sorry', 'thankyou'])

file_name = 'big_pose_2nd'

model = islr()

model.load_weights(f'islr_{file_name}.h5')

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_hol.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()

        img, results = colour_conv(frame, holistic)
        
        landmark_show(img, results)
        
        keypoints = extract_pose_hand_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))
            print(actions[np.argmax(res)])
            
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            if len(sentence) > 0: 
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
            else:
                sentence.append(actions[np.argmax(res)])
            
            cv2.rectangle(img, (0,0), (200, 90), (0, 0, 0), -1)
            cv2.putText(img, sentence[-1], (20,80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) 
          
        cv2.imshow('ISLR Real-Time Prediction', img)
        cv2.namedWindow('ISLR Real-Time Prediction', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ISLR Real-Time Prediction', 800, 600)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

'''
The code below is the one I used to test the model on a video. I used the mediapipe library to extract the landmarks. I then used the model to predict the sign language and displayed the prediction on the screen.
How can I edit the code to make it capture 30 frames, predict the sign and display the prediction on the screen, and then only after pressing a certain key, repeat the process for the next 30 frames instead of constantly predicting the last 30 frames?
'''