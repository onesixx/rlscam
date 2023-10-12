import multiprocessing
multiprocessing.cpu_count()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. set hyperparameters
batch_size = 256
learning_rate = 0.0002
num_epoch = 10

# 2. Data
classes = {'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

# 2-1 Download Data
cifar_trn = dset.CIFAR10("./", train=True,  
    transform=transforms.ToTensor(), target_transform=None, download=True)
cifar_tst  = dset.CIFAR10("./", train=False, 
    transform=transforms.ToTensor(), target_transform=None, download=True)
# 2-2 check Dataset
# cifar_trn.__getitem__(0)[0].size(), cifar_trn.__len__(),   list(classes)[ cifar_trn.__getitem__(0)[1] ]
# cifar_tst.__getitem__(0)[0].size(), cifar_tst.__len__(),  cifar_trn.__getitem__(0)[1]

import matplotlib.pyplot as plt
for i in range(6):
    img = cifar_trn.__getitem__(i)[0].numpy().transpose(1,2,0)
    plt.imshow(img)
    plt.show()

# import plotly.express as px
# import plotly.graph_objects as go
# for i in range(6):
#     fig = go.Figure()
#     img = cifar_trn.__getitem__(i)[0].numpy().transpose(1, 2, 0)
#     fig.add_trace(go.Image(z=img))
#     fig.show()

# 2-3 Set DataLoader
trn_loader = torch.utils.data.DataLoader( cifar_trn, 
    batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)  # 24개의 쓰레드 사용
tst_loader = torch.utils.data.DataLoader( cifar_tst, 
    batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True) 

# 3. Model & Optimizer (N개, Channel, Height, Width)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(           #  (3,32,32)
            nn.Conv2d(3, 16, 3, padding=1),   # (16, 32, 32(=32+2 -(3-1)))
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),  # (32, 32, 32(=32+2 - (3-1)))
            nn.ReLU(),
            nn.MaxPool2d(2,2),                # (32, 16, 16(=32/2))
            nn.Conv2d(32, 64, 3, padding=1 ), # (64, 16 ,16(=16+2-(3-1)))
            nn.ReLU(),
            nn.MaxPool2d(2,2),                 # (128, 8, 8(=16/2)) 

            nn.Conv2d(64,128,3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128,256,3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)                 # (256, 4, 4(=8/2))
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(256*4*4, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        )
    
    def forward(self, x):
        out = self.layer(x)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)
        return out

model = CNN()

# 3-1 Loss func & Optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 4. Train
import time
start = time.time()

for i in range(num_epoch):
    for j, [image, label] in enumerate(trn_loader):
        x = image
        y_= label

        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output, y_)
        loss.backward()
        optimizer.step()

        if j % 100== 0:
            print(f'epoch: {i} , train: {j}, loss: {loss}, loss.data: {loss.data}')
print("Done !!!")

param_list = list(model.parameters())
print(param_list)

# 5. Test
total = 0
correct = 0

for image, label in tst_loader:
    x = image
    y_= label

    output = model.forward(x)
    _ , output_index = torch.max(output,1)
    
    total += label.size(0)
    correct += (output_index == y_).sum().float()

print(f"Accuracy of Test Data: {100*correct/total}")