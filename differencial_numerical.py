import numpy as np
import plotly.graph_objs as go

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fun2(X):
    return X[0]**2 + X[1]**2

x0 = np.linspace(-3, 3,  30)
x1 = np.linspace(-3, 3,  30)
x0, x1 = np.meshgrid(x0, x1)

y = np.zeros((len(x0), len(x1)))
X = np.zeros(2)
xn = x0.shape[0]  # 30

for i1 in range(xn):
    for i2 in range(xn):
        X[0] = x0[i1, i2]
        X[1] = x1[i1, i2]
        y[i1, i2] = fun2(X)
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x0, x1, y, 
    # cmap='viridis',
    rstride=1, cstride=1, 
    alpha=0.3, color='white', edgecolor='black')
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('y')
plt.show()

fig = go.Figure(data=[go.Surface(x=x0, y=x1, z=y)])
fig.update_layout(title='Half Sphere', autosize=False,
    width=500, height=500,
    margin=dict(l=65, r=50, b=65, t=90),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    showlegend=False)  # 오른쪽 legend 숨기기
fig.show()