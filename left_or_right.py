import os
import numpy as np
import shutil

left_hand_folders = []
right_hand_folders = []
both_hands_folders = []
null_folders = []
folder_count = 0

path_hand = '/help_duplicate/signs_hands/hi'
path_pose = '/help_duplicate/signs_pose/hi'

for folder in os.listdir(path_hand):
    folder_path = os.path.join(path_hand, folder)
    folder_count += 1
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file.endswith(".npy"):
                sequence = np.load(file_path)
                if np.array_equal(sequence[-126:], np.zeros(63*2)):
                    null_folders.append(folder)
                if np.count_nonzero(sequence) > 64:
                    both_hands_folders.append(folder)
                elif np.array_equal(sequence[-63:], np.zeros(63)):
                    left_hand_folders.append(folder)
                elif np.array_equal(sequence[:63], np.zeros(63)):
                    right_hand_folders.append(folder)

left_hand_folders = list(dict.fromkeys(left_hand_folders))
right_hand_folders = list(dict.fromkeys(right_hand_folders))
null_folders = list(dict.fromkeys(null_folders))
both_hands_folders = list(dict.fromkeys(both_hands_folders))

left_hand_folders.sort()
right_hand_folders.sort()
null_folders.sort()
both_hands_folders.sort()

inter = list(set(left_hand_folders) & set(right_hand_folders))
inter.sort()

# print("Left hand folders: ", left_hand_folders)
# print("Right hand folders: ", right_hand_folders)
print("Both hands folders: ", both_hands_folders)
print("Intersection: ", inter)

for folder in inter:
    if folder in left_hand_folders:
        left_hand_folders.remove(folder)
    if folder in right_hand_folders:
        right_hand_folders.remove(folder)

majority = []
for folder in inter:
    folder_path = os.path.join(path_hand, folder)
    left_count = 0
    right_count = 0
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".npy"):
            sequence = np.load(file_path)
            if np.array_equal(sequence[-63:], np.zeros(63)):
                left_count += 1
            elif np.array_equal(sequence[:63], np.zeros(63)):
                right_count += 1
    if left_count > right_count:
        majority.append("left")
        left_hand_folders.append(folder)
    elif right_count > left_count:
        majority.append("right")
        right_hand_folders.append(folder)
    else:
        majority.append("equal")

print("Majority in intersection: ", majority)

for folder in left_hand_folders:
    if folder in null_folders:
        null_folders.remove(folder)
    if folder in both_hands_folders:
        left_hand_folders.remove(folder)

for folder in right_hand_folders:
    if folder in null_folders:
        null_folders.remove(folder)
    if folder in both_hands_folders:
        right_hand_folders.remove(folder)
    


left_hand_folders = list(dict.fromkeys(left_hand_folders))
right_hand_folders = list(dict.fromkeys(right_hand_folders))
null_folders = list(dict.fromkeys(null_folders))

left_hand_folders.sort()
right_hand_folders.sort()


print("Left hand folders: ", left_hand_folders)
print(len(left_hand_folders))
print("Right hand folders: ", right_hand_folders)
print(len(right_hand_folders))
print(folder_count)
print('Both hands folders: ', both_hands_folders)
print(len(both_hands_folders))
print("Null folders: ", null_folders)


os.makedirs(os.path.join(path_pose, "left"))
os.makedirs(os.path.join(path_pose, "right"))

# move left hand folders to left folder
for folder in left_hand_folders:
    folder_path_old = os.path.join(path_pose, folder)
    folder_path_new = os.path.join(path_pose, "left", folder)
    shutil.move(folder_path_old, folder_path_new)

# move right hand folders to right folder
for folder in right_hand_folders:
    folder_path_old = os.path.join(path_pose, folder)
    folder_path_new = os.path.join(path_pose, "right", folder)
    shutil.move(folder_path_old, folder_path_new)