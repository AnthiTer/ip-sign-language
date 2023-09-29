import numpy as np
import os
import plotly.express as px

# DATA_PATH = os.path.join('/duplicate/signs')
DATA_PATH_HAND = os.path.join('/duplicate/signs_hands')
DATA_PATH_POSE = os.path.join('/duplicate/signs_pose')

# signs = np.array(['goodbye', 'goodmorning', 'hi', 'me', 'thankyou', 'you'])

# signs = np.array(['come', 'good', 'happy', 'home', 'iloveyou', 'sorry'])

signs = np.array(['come', 'good', 'goodbye', 'goodmorning', 'happy', 'hi', 'home', 'iloveyou', 'me', 'sorry', 'thankyou', 'you'])

# signs = np.array(['come', 'good', 'goodbye', 'goodmorning', 'happy', 'hi', 'home', 'iloveyou', 'sorry', 'thankyou'])

counters = np.zeros(len(signs))
for s in signs:
    dir_seq = os.path.join(DATA_PATH_POSE, s)
    folders = os.listdir(dir_seq)
    for fold in folders:
        counters[signs.tolist().index(s)] += 1
print(counters)
print(np.sum(counters))

fig = px.bar(x=signs, y=counters)
fig.update_layout(title='Video Distribution', plot_bgcolor='white',legend_title='signs')
fig.update_xaxes(title='Signs', showgrid=False)
fig.update_yaxes(title='Count', showgrid=False)
fig.show()

