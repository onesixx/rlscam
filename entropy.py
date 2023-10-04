import numpy as np
import plotly.graph_objs as go

def entropy(x):
    return -x * np.log(x)

x = np.linspace(0, 1.0, 50)
y = entropy(x)

fig= go.Figure() # trace, layout
fig.add_trace(
    go.Scatter(
        x=x, y=y, mode='lines+markers',
        line=dict(color='blue'),
        marker=dict(size=4, color='red')
    ))
fig.update_layout(
   title = f"{entropy.__name__} Plot"
)
fig.show()

## plot
trace = go.Scatter(
    x=x, y=y, mode='lines+markers',
    line=dict(color='blue'),
    marker=dict(size=4, color='red')
)
layout = go.Layout(
   title = f"{entropy.__name__} Plot", 
    xaxis=dict(range=[0, 1], scaleanchor='x', scaleratio=1, title='x'),
    yaxis=dict(range=[0, 1]),
    width=500
)
fig = go.Figure( data=[trace], layout=layout)
fig.show()