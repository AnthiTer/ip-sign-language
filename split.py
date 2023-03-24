from sklearn.model_selection import train_test_split
import numpy as np

x = np.load('seq_labels/sequences_pose_all.npy')
y = np.load('seq_labels/labels_pose_all.npy')

xtrain, xrest, ytrain, yrest = train_test_split(x, y, test_size=0.3, random_state=42)
xval, xtest, yval, ytest = train_test_split(xrest, yrest, test_size=0.5, random_state=42)

np.save('data_all/xtrain.npy',xtrain)
np.save('data_all/xval.npy',xval)
np.save('data_all/xtest.npy',xtest)
np.save('data_all/ytrain.npy',ytrain)
np.save('data_all/yval.npy',yval)
np.save('data_all/ytest.npy',ytest)