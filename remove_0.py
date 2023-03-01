import os
import shutil
import numpy as np
import plotly.express as px

# Set the path to the bigger folder
path = '/Users/anthi/Documents/IP/signs_hands/happy'
path1 = '/Users/anthi/Documents/IP/signs_pose/happy'
file = '0.npy'
list = []
no_remove = 0
list_names=[]

def find_outliers_iqr(data, kl=0.35):
    q1, q2, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * kl)
    return lower_bound, q2

for folder in os.listdir(path):
    folder_pathname = os.path.join(path, folder)

    if not os.path.isdir(folder_pathname):
        continue
    
    npy_files = [filename for filename in os.listdir(folder_pathname) if filename.endswith('.npy')]
    if len(npy_files) == 0:
        continue
    
    ref_array = np.load(os.path.join(file))
    ref_count = 30
    
    for npy_filename in npy_files:
        npy_filepath = os.path.join(folder_pathname, npy_filename)
        npy_array = np.load(npy_filepath)
        
        # Check if the array is the same as the reference array
        if np.array_equal(npy_array, ref_array):
            ref_count -= 1
    

    list.append(ref_count)
    # Check if the reference count is greater than 25
    # if ref_count > 23:
    #     print(f"Folder {folder} contains {ref_count} non-null npy files.")
    #     no_remove +=1
    
    if ref_count < 8:
        print(f"Folder {folder} contains {ref_count} non-null npy files.")
        no_remove +=1
        list_names.append(folder)

print(sum(list)/120)
print(find_outliers_iqr(list))
print(no_remove)
print(list_names)

fig = px.histogram(list, nbins = 50)
fig.show()

for f in list_names:
    folder_path = os.path.join(path1, f)
    if os.path.exists(folder_path):
        # shutil.rmtree(folder_path)
        print(f"delete {folder_path}")