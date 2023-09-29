import numpy as np
import os

# DATA_PATH = os.path.join('/duplicate/signs')
DATA_PATH_HAND = os.path.join('/duplicate/signs_hands')
DATA_PATH_POSE = os.path.join('/duplicate/signs_pose')

# signs = np.array(['goodbye', 'goodmorning', 'hi', 'me', 'thankyou', 'you'])

# signs = np.array(['come', 'good', 'happy', 'home', 'iloveyou', 'sorry'])

# signs = np.array(['come', 'good', 'goodbye', 'goodmorning', 'happy', 'hi', 'home', 'iloveyou', 'me', 'sorry', 'thankyou', 'you'])

signs = np.array(['come', 'good', 'goodbye', 'goodmorning', 'happy', 'hi', 'home', 'iloveyou', 'sorry', 'thankyou'])

label_map = {label:num for num, label in enumerate(signs)}

sequences = []
labels = []

def label_data():
    for s in signs:
        dir_seq = os.path.join(DATA_PATH_POSE, s)
        folders = os.listdir(dir_seq)
        for fold in folders:
            path_vid = os.path.join(dir_seq, fold)
            if not os.path.isdir(path_vid):
                continue
            frame_files = os.listdir(path_vid)
            frame_files.sort(key=lambda x: int(x[:-4]))
            
            # create an empty window to store the frames for the current sequence
            window = []
            
            # loop through all frame files for the current sequence
            for ff in frame_files:
                
                # load the current frame and append it to the window
                frame_path = os.path.join(path_vid, ff)
                frame = np.load(frame_path)
                window.append(frame)
            sequences.append(window)
            labels.append(label_map[s])

label_data()
print(np.array(sequences).shape)
print(np.array(labels).shape)


np.save('sequences_pose_nomy.npy', np.array(sequences))
np.save('labels_pose_nomy.npy', np.array(labels))