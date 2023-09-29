import os
import shutil
import numpy as np
import plotly.express as px

'''
From the folder signs_hands, classify the sequence of signs into 2 groups: left and right hand. 
There are many folders, which each of them invclude a sequence of 30 frames in npy form.
The left hand is the one that is ending with 63 0's and the right hand is the one that is starting with 63 0s.
How can I add the folder name to the bigger folders of the left hand or the right hand?
'''

# me = 0.45
# you = 0.55
# home = 1.4
# come - no need to remove anything, collection went well
# good = 1.2
# goodbye = 1.5
# goodmorning = 0.7 ???
# happy = 1.3
# hi = 1
# iloveyou = 1.2
# sorry - no need to remove anything, collection went well
# thankyou = 1.2

# Set the path to the bigger folder
path = '/duplicate/signs_hands/me'
path1 = '/duplicate/signs_pose/me'
file = '0.npy'
list = []
no_remove = 0
list_names=[]
list = []
file_list = []
folder_names = []
count = 0

def find_outliers_iqr(data, kl=0.45):
    q1, q2, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * kl)
    upper_bound = q3 + (iqr * kl)
    return lower_bound, upper_bound

for folder in os.listdir(path):
    folder_pathname = os.path.join(path, folder)

    if not os.path.isdir(folder_pathname):
        continue
    
    npy_files = [filename for filename in os.listdir(folder_pathname) if filename.endswith('.npy')]
    if len(npy_files) == 0:
        continue
    
    ref_array = np.load(os.path.join(file))
    ref_count = 30

    count += 1
    
    for npy_filename in npy_files:
        npy_filepath = os.path.join(folder_pathname, npy_filename)
        npy_array = np.load(npy_filepath)
        
        # Check if the array is the same as the reference array
        if np.array_equal(npy_array, ref_array):
            ref_count -= 1
    
    list.append(ref_count)
    file_list.append(npy_files)
    folder_names.append(folder)

# Call find_outliers_iqr function with list list
lower_bound, upper_bound = find_outliers_iqr(list)

# Check if each folder count is an outlier or not
for i in range(len(list)):
    if list[i] > 25 or list[i] < 7:
    # if list[i] < 12:
        print(f"Folder {folder_names[i]} contains {list[i]} non-null npy files.")
        no_remove +=1
        list_names.append(folder_names[i])

print(sum(list)/count)
print(lower_bound, upper_bound)
print(no_remove)
print(list_names)
print(count)

fig = px.histogram(list, nbins = 30)
fig.show()

fold = [path, path1]

for f in list_names:
    for p in fold:
        folder_path = os.path.join(p, f)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"delete {folder_path}")