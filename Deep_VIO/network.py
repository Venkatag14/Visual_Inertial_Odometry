import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from turtle import forward
from torchsummary import summary
# import lightning as L


class VoNet(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        
        self.vonet = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=7, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 128, kernel_size=5, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 256, kernel_size=5, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 512, kernel_size=3, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 1024, kernel_size=3, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(1024, 1024, kernel_size=3, stride=1),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Flatten())
        
        self.q_vonet = nn.Sequential(nn.Linear(9216, 512),
                                     nn.Tanh(),
                                     nn.Linear(512, 512),
                                     nn.Tanh(),
                                     nn.Linear(512, 4))
        
        self.t_vonet = nn.Sequential(nn.Linear(9216, 512),
                                     nn.Tanh(),
                                     nn.Linear(512, 512),
                                     nn.Tanh(),
                                     nn.Linear(512, 3))
                                     
                                     
    
    def forward(self, input):
        
        flat_output = self.vonet(input)
        q_output = self.q_vonet(flat_output)
        t_output = self.t_vonet(flat_output)
        
        return q_output, t_output
        
        # #normalize quaternions
        # i, j, k = q_output[1, 1], q_output[1, 2], q_output[1, 3]
        
        # magnitude = torch.sqrt(i**2 + j**2 + k**2)
        
        # i /= magnitude
        # j /= magnitude
        # k /= magnitude
        
        # q_output[:, 1] = i
        # q_output[0, 2] = j
        # q_output[0, 3] = k
        
class IoNet(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, num_classes=4):
        super(IoNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layers
        self.fc_q = nn.Sequential(
            nn.Linear(hidden_size * 10, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )
        
        self.fc_t = nn.Sequential(
            nn.Linear(hidden_size * 10, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
    
    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Reshape output to (batch_size, hidden_size * seq_length)
        out = out.reshape(out.size(0), -1)
        
        # Fully connected layer
        q_output = self.fc_q(out)
        t_output = self.fc_t(out)
        
        return q_output, t_output
    
    

    
class VIoNet(nn.Module):
    def __init__(self, input_channels =2, input_size=6, hidden_size=64, num_layers=2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.vonet = nn.Sequential(nn.Conv2d(input_channels, 64, kernel_size=7, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 128, kernel_size=5, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 256, kernel_size=5, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 512, kernel_size=3, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 512, kernel_size=3, stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(512, 1024, kernel_size=3, stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(1024, 1024, kernel_size=3, stride=1),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Flatten())
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.q_vionet = nn.Sequential(nn.Linear(9216+hidden_size*10, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 4))
        
        self.t_vionet = nn.Sequential(nn.Linear(9216+hidden_size*10, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 3))
        
    def forward(self, images, IMU):
        
        h0 = torch.zeros(self.num_layers, IMU.size(0), self.hidden_size).to(IMU.device)
        c0 = torch.zeros(self.num_layers, IMU.size(0), self.hidden_size).to(IMU.device)
        
        
        
        img_flat_output = self.vonet(images)
        imu_flat_output, _ = self.lstm(IMU, (h0, c0))
        
        imu_flat_output = imu_flat_output.reshape(imu_flat_output.size(0), -1)
        
        flat_output = torch.cat((img_flat_output, imu_flat_output), dim=1)
        
        q_output = self.q_vionet(flat_output)
        t_output = self.t_vionet(flat_output)
        
        return q_output, t_output
    
  

# vionet = VIoNet(2)

# input_shape1 = (2, 512, 512)
# input_shape2 = (6, 100)

# input_shapes = [input_shape1, input_shape2]

# input1 = torch.randn(5, *input_shape1)
# input2 = torch.randn(5, *input_shape2)

# # summary(vionet, input_shapes)

# q, t = vionet(input1, input2)

# print(q, t)

# vionet = VIoNet().to('cuda')

# input_shape2 = (100,6)
# input_shape1 = (2, 512, 512)

# input1 = torch.randn(5, *input_shape1).to('cuda')
# input2 = torch.randn(5, *input_shape2).to('cuda')

# q,t = vionet(input1, input2)

# print(q, t)







                                 
                                   
                                   
                                   
                                   