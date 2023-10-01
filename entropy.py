import numpy as np
import plotly.graph_objs as go

def entropy(x):
    return -np.sum(x * np.log2(x))

x = np.arange(0.1, 10,.3)
   #np.linspace(-5.0, 5.0, 100)

y = entropy(x)

fig= go.Figure() # trace, layout
fig.add_trace(
    go.Scatter(x=x, y=y, mode='lines+markers',
               line=dict(color='blue'),
               marker=dict(size=4, color='red')))
fig.update_layout(
    title = f"{entropy.__name__} Plot", 
    #yaxis_range=[-0.1, 1.1] # y축의 범위 지정
)