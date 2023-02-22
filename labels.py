import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch

# DATA_PATH = os.path.join('/Users/anthi/Documents/IP/duplicate/signs')
DATA_PATH_HAND = os.path.join('/Users/anthi/Documents/IP/duplicate/signs_hands')
DATA_PATH_POSE = os.path.join('/Users/anthi/Documents/IP/duplicate/signs_pose')

no_vid = 30
frames = 30
start = 0

actions = np.array(['goodbye', 'hi', 'you', 'me', 'thankyou', 'goodmorning'])

label_map = {label:num for num, label in enumerate(actions)}

sequences = []
labels = []

def label_data():
    for action in actions:
        sequences_dir = os.path.join(DATA_PATH_HAND, action)
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

# np.save('sequences.npy', sequences)
# np.save('labels.npy', labels)

# sequences = np.load('sequences.npy')
# labels = np.load('labels.npy')

X = np.array(sequences)
# labels_tensor = (labels, dtype=torch.int64).clone().detach()
# y = torch.nn.functional.one_hot(torch.tensor(labels_tensor)).to(torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.int64)
labels_tensor_copy = labels_tensor.clone().detach()
y = torch.nn.functional.one_hot(labels_tensor_copy).to(torch.float32)

print(y)