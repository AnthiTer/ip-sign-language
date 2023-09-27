import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# data = pd.read_csv('error_bars_lstm.csv')
data = pd.read_csv('error_bars_gru.csv')
signs = data.columns

means = data.mean()
stds = data.std()

df = pd.DataFrame({'signs': means.index, 'means': means.values, 'stds': stds.values})

print(df)

# Create the bar plot with error bars
# fig = px.bar(df, x='signs', y='means', error_y='stds')

# # Update layout and axis labels
# fig.update_layout(title='Error Bars', plot_bgcolor='white',legend_title='Signs')
# fig.update_xaxes(title='Signs', showgrid=False)
# fig.update_yaxes(range=[0.6, 1.02], tick0=0.7, dtick=0.05)
# fig.update_yaxes(title='F1 Score', showgrid=False)


# # Show the plot
# fig.show()



# df = pd.DataFrame({'metrics': means.index, 'means': means.values, 'stds': stds.values})

# # Create the bar plot with error bars
# fig = px.bar(df, x='metrics', y='means', error_y='stds')

# # Update layout and axis labels
# fig.update_layout(title='Error Bars', plot_bgcolor='white',legend_title='metrics')
# fig.update_xaxes(title='Metrics', showgrid=False)
# fig.update_yaxes(range=[0.6, 1.02], tick0=0.7, dtick=0.05)
# fig.update_yaxes(title='Value', showgrid=False)

# # Show the plot
# fig.show()

import plotly.graph_objects as go

# Create bar trace
# bar_trace = go.Bar(x=df['signs'], y=df['means'], error_y=dict(type='data', array=df['stds']))

# Create new trace for error bars
error_trace = go.Scatter(x=df['signs'], y=df['means'], mode='markers', error_y=dict(type='data', array=df['stds']), marker=dict(color='black'))

# Create figure with both traces
fig = go.Figure(data=[error_trace])

# Update layout and axis labels
fig.update_layout(title='Error Bars', plot_bgcolor='white', legend_title='Signs')
fig.update_xaxes(title='Signs', showgrid=False)
fig.update_yaxes(range=[0.6, 1.02], tick0=0.7, dtick=0.05)
fig.update_yaxes(title='F1 Score', showgrid=False)

# Show the plot
fig.show()