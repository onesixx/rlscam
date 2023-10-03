import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def func_2(X):
    return X[0]**2 + X[1]**2

def _numerial_gradient_no_batch(f,x):
    # 수치 미분 ㅊ  
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = float(tmp_val) + h
        fxh1 = f(x)
        x[idx] = tmp_val -h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
        print(x[idx], grad[idx])
    return grad

def numerical_gradient(f,X):
    if X.ndim ==1:
        return _numerial_gradient_no_batch(f,X)
    else:
        grad = np.zeros_like(X)
        for idx, x in enumerate(X):
            grad[idx] = _numerial_gradient_no_batch(f,x)
        return grad

x0= np.arange(-2, 2, 0.25)
x1= np.arange(-2, 2, 0.25)
X, Y = np.meshgrid(x0, x1)

X = X.flatten()
Y = Y.flatten()
grad = numerical_gradient(func_2, np.array([X, Y]).T ).T 

plt.figure()
plt.quiver(X, Y, -grad[0], -grad[1], angles="xy",color="#666666")
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('x0')
plt.ylabel('x1')
plt.grid()
plt.legend()
plt.draw()
plt.show()
