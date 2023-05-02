import numpy as np
# import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

x = np.linspace(-2.5, 2.5, 100)

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def prelu(x, alpha=0.1):
    return np.where(x >= 0, x, alpha*x)

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha*(np.exp(x)-1))

def leaky_relu(x, alpha=0.1):
    return np.maximum(alpha * x, x)

# Plot activation functions
# plt.figure(figsize=(10, 6))

# plt.plot(x, sigmoid(x), label='sigmoid')
# plt.plot(x, relu(x), label='ReLU')
# plt.plot(x, tanh(x), label='tanh')
# plt.plot(x, softmax(x), label='softmax')
# plt.plot(x, prelu(x), label='PReLU')
# plt.plot(x, elu(x), label='ELU')
# plt.plot(x, leaky_relu(x), label='Leaky ReLU')

# plt.legend()
# plt.title('Activation Functions')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

df = pd.DataFrame({'x': x,
                'sigmoid': sigmoid(x),
                 'relu': relu(x),
                 'tanh': tanh(x),
                 'leaky_relu': leaky_relu(x)})

# Create the plot using Plotly Express
fig = px.line(df, x='x', y=['sigmoid', 'relu', 'tanh', 'leaky_relu'],
              labels={'value': 'y', 'variable': 'Activation Function'},
              title='Activation Functions')
fig.write_image('graphs/activation_functions.png')
fig.show()
