# https://youtu.be/KgH3ZWmMxLE?si=vPMo4vdyjWGd9pDI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px

X = np.random.rand(100)
Y = .2 * X + .5

plt.figure()
plt.scatter(X, Y)
plt.show()


# https://wikidocs.net/book/8909
common_layout = go.Layout(
    width =500, height=500, template = 'plotly_white',
    xaxis=dict(title='X 축'),
    yaxis=dict(title='Y 축'),
    title='Title'
)

fig = go.Figure(
    data= [ go.Scatter(x=X, y=Y, mode='markers')],
    layout=common_layout)
fig.update_layout(title="Scatter Plot", 
    xaxis=dict(range=[0,1]),
    yaxis=dict(range=[0,1]),
)
fig.show()

df = pd.DataFrame({'X':X, 'Y':Y})
fig = px.scatter(df, x=X, y=Y)
fig.update_layout(common_layout)
fig.update_layout(title="Scatter Plot", 
    xaxis=dict(range=[0,1]),
    yaxis=dict(range=[0,1]),
)
fig.show()

def plot_prediction0(pred, y):
    plt.figure()
    plt.scatter(X, Y)
    plt.scatter(X, pred)
    plt.show()

def plot_prediction(pred, y):
    fig = go.Figure(
        data= [ go.Scatter(x=X, y=Y, mode='markers'),
                go.Scatter(x=X, y=pred, mode='markers')],
        layout=common_layout)
    fig.update_layout(title="Scatter Plot", 
        xaxis=dict(range=[0,1]),
        yaxis=dict(range=[0,1]),
    )
    fig.show()

W = np.random.uniform(-1,1)
b = np.random.uniform(-1,1)
learning_rate = 0.7

for epoch in range(100):
    pred = W * X + b
    error = np.abs(Y - pred).mean()
    if error < 0.0001:
        break
    # gradient descent
    W_grad = learning_rate * ((pred-Y)*X).mean()  # learning_rate * np.dot(error, X)
    b_grad = learning_rate * (pred - Y).mean()    # learning_rate * error.mean()

    # update
    W = W - W_grad
    b = b - b_grad

    if epoch % 10 == 0:
        pred = W * X + b
        print(f"{epoch:2} W={W:.5f}, b={b:.5f}, error={error.mean():.5f}")      
        plot_prediction(pred, Y)


