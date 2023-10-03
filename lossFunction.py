import numpy as np

def mse(y, t):  # Mean Squared Error
    return 0.5 * np.sum((y-t)**2)

def cee(y, t):  # Cross Entropy Error
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

yt = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) # 정답은 '2'
# softmax의 출력
y0 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) 
y1 = np.array([.1, .05, .6, 0, .05, .1, 0,  .1, 0, 0]) # 2일 확률이 가장 높다고 추정함
y2 = np.array([.1, .05, .1, 0, .05, .1, 0,  .6, 0, 0]) # 7일 확률이 가장 높다고 추정함
y3 = np.array([.1, .1, .1,.1, .1, .1, .1, .1, .1, .1])                  #  의미없는 값    
y4 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])                     # 완전 틀림
y5 = np.array([.11, .11, 0, .11, .11, .11, .11, .11, .11, .12]) # 완전 틀림

#   작을수록 정답에 가까움
mse(y0, yt) # 0.00
mse(y1, yt) # 0.09
mse(y2, yt) # 0.59
mse(y3, yt) # 0.45
mse(y4, yt) # 1.00
mse(y5, yt) # 0.55

cee(y0, yt) # -9.99e-08 near 0
cee(y1, yt) # 0.51
cee(y2, yt) # 2.30
cee(y3, yt) # 2.30
cee(y4, yt) # 16.11
cee(y5, yt) # 16.11
