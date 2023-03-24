import numpy as np
import os

# DATA_PATH = os.path.join('/Users/anthi/Documents/IP/duplicate/signs')
DATA_PATH_HAND = os.path.join('/Users/anthi/Documents/IP/duplicate/signs_hands')
DATA_PATH_POSE = os.path.join('/Users/anthi/Documents/IP/duplicate/signs_pose')

# actions = np.array(['goodbye', 'goodmorning', 'hi', 'me', 'thankyou', 'you'])

actions = np.array(['come', 'good', 'happy', 'home', 'iloveyou', 'sorry'])

# actions = np.array(['come', 'good', 'goodbye', 'goodmorning', 'happy', 'hi', 'home', 'iloveyou', 'me', 'sorry', 'thankyou', 'you'])

label_map = {label:num for num, label in enumerate(actions)}

sequences = []
labels = []

def label_data():
    for action in actions:
        sequences_dir = os.path.join(DATA_PATH_POSE, action)
        sequence_folders = os.listdir(sequences_dir)
        for sequence_folder in sequence_folders:
            sequence_path = os.path.join(sequences_dir, sequence_folder)
            if not os.path.isdir(sequence_path):
                continue
            frame_files = os.listdir(sequence_path)
            frame_files.sort(key=lambda x: int(x[:-4]))
            
            # create an empty window to store the frames for the current sequence
            window = []
            
            # loop through all frame files for the current sequence
            for frame_file in frame_files:
                
                # load the current frame and append it to the window
                frame_path = os.path.join(sequence_path, frame_file)
                frame = np.load(frame_path)
                window.append(frame)
            sequences.append(window)
            labels.append(label_map[action])

label_data()
print(np.array(sequences).shape)
print(np.array(labels).shape)


np.save('sequences_pose_2nd.npy', np.array(sequences))
np.save('labels_pose_2nd.npy', np.array(labels))