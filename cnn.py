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
batch_size = 100
learning_rate = 0.002
num_epoch = 10

# 2. Data
# 2-1 Download Data
mnist_train = dset.MNIST("./", train=True,  
    transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test  = dset.MNIST("./", train=False, 
    transform=transforms.ToTensor(), target_transform=None, download=True)
# 2-2 check Dataset
# mnist_train.__getitem__(0)[0].size(), mnist_train.__len__()
# mnist_test.__getitem__(0)[0].size(), mnist_test.__len__()

# 2-3 Set DataLoader
train_loader = torch.utils.data.DataLoader( mnist_train, 
    batch_size=batch_size, shuffle=True, num_workers=24, drop_last=True)  # 24개의 쓰레드 사용
test_loader = torch.utils.data.DataLoader( mnist_test, 
    batch_size=batch_size, shuffle=False, num_workers=24, drop_last=True) 

# 3. Model & Optimizer (N개, Channel, Height, Width)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential( # initial 이미지 (1,28,28)
            nn.Conv2d(1, 16, 5),  # output 이미지(16, 24(=28-(5-1)),24); for edge =>( in채널:1흑백, out채널:16(필터갯수), 커널크기:5, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            nn.ReLU(),
            nn.Conv2d(16, 32, 5), # output 이미지(32, 20(=24-(5-1), 20);  for texture 
            nn.ReLU(),
            nn.MaxPool2d(2,2),    # output 이미지(32, 20(=20/2), 10);  for Featuer Reduction/ robust:  kernel_size=2, stride=2
            nn.Conv2d(32, 64, 5), # output 이미지(64, 6(=10-(5-1)), 6); for object, chennel=32, out_channel=64, kernel_size=5
            nn.ReLU(),
            nn.MaxPool2d(2,2)     # output 이미지(64, 3(=6/2), 3) 
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64*3*3, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
    
    def forward(self, x):
        out = self.layer(x)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)
        return out

model = CNN()

# 3-1 Loss func & Optimizer
loss_func = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 4. Train
for i in range(num_epoch):
    for j, [image, label] in enumerate(train_loader):
        x = image
        y_= label

        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output, y_)
        loss.backward()
        optimizer.step()

        if j % 100== 0:
            print(f'epoch: {i} , train: {j}, loss: {loss}')
print("Done !!!")

# 5. Test
total = 0
correct = 0

for image, label in test_loader:
    x = image
    y_= label

    output = model.forward(x)
    _ , output_index = torch.max(output,1)
    
    total += label.size(0)
    correct += (output_index == y_).sum().float()

print(f"Accuracy of Test Data: {100*correct/total}")