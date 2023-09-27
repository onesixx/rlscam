import numpy as np
import plotly.graph_objs as go

def step_function(x):
    return np.array(x > 0, dtype=np.int32)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def identity_function(x):
    return x
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.arange(-5.0, 5.0,.3)
x = np.linspace(-1.0, 1.0,500)

y = step_function(x)
y = sigmoid(x)
y = relu(x)
y = identity_function(x)
y = softmax(x)

fig= go.Figure() # trace, layout
fig.add_trace(
    go.Scatter(x=x, y=y, mode='lines+markers',
               line=dict(color='blue'),
               marker=dict(size=4, color='red')))
fig.update_layout(
    title = f"{relu.__name__} Plot", 
    xaxis_range=[-2, 2] # y축의 범위 지정
)
fig.show()



