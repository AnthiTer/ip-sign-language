import os

root_folder = '/Users/anthi/Documents/IP/signs_hands'  # change this to the path of your root folder

for foldername, subfolders, filenames in os.walk(root_folder):
    num_files = len(filenames)
    if num_files == 30:
        next
    else:
        print(f"The folder {foldername} does not have 30 files. It has {num_files} files.")