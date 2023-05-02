import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

history = pd.read_csv('graphs/history_all_gru_please_help_me.csv')

# plt.figure(figsize=(8, 6))
# plt.plot(history['categorical_accuracy'], label='Training Accuracy', color='blue')
# plt.plot(history['val_categorical_accuracy'], label='Validation Accuracy', color='orange')
# plt.xlabel('Epochs', fontsize=14)
# plt.ylabel('Accuracy', fontsize=14)
# plt.title('Categorical Accuracy', fontsize=16)
# plt.legend()
# plt.axhline(y=1, color='black', linestyle='--')
# # plt.savefig(f'graphs/accuracy_{file_name}.png')
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.plot(history['loss'], label='Training Loss', color='green')
# plt.plot(history['val_loss'], label='Validation Loss', color='red')
# plt.xlabel('Epochs', fontsize=14)
# plt.ylabel('Loss', fontsize=14)
# plt.title('Training and Validation Loss', fontsize=16)
# plt.legend()
# # plt.savefig(f'graphs/accuracy_{file_name}.png')
# plt.show()

fig = px.line(history, x=history.index, y=['categorical_accuracy', 'val_categorical_accuracy'],
              labels={'value': 'Accuracy', 'variable': 'Dataset', 'index': 'Epochs'},
              title='Training and Validation Accuracy')
fig.update_layout(xaxis_title='Epochs', yaxis_title='Accuracy',
                  yaxis_range=[0, 1.02],legend_title=None)
newnames = {'categorical_accuracy': 'Training Accuracy', 'val_categorical_accuracy': 'Validation Accuracy'}
fig.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     )
                  )
fig.show()