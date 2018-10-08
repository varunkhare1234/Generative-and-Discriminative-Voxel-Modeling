import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.optim as optim
import pandas as pd
import torchvision.models as models_res
import matplotlib.pyplot as plt
from skimage import io, transform
    
def weights_init(m):
    ##later call net = voxception(); net.apply(weights_init) to recursively set init
    if isinstance(m, nn.Conv3d) or isinstance(m,nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain = nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
#   explicitly set the init of these two layers to gain=1 later
#         nn.init.orthogonal_(net.input_conv.weight)
#         nn.init.orthogonal_(net.fc[-1].weight)
 
class Voxception_Downsample_Layer(nn.Module):
    def __init__(self,numChannels,batch_norm=False,elu=False):
        super(Voxception_Downsample_Layer, self).__init__()
        if (not elu) and (not batch_norm):
            self.branch0 = nn.Conv3d(numChannels, int(numChannels/2), kernel_size=3, stride=2,padding=1)
            self.branch1 = nn.Conv3d(numChannels, int(numChannels/2), kernel_size=1, stride=2)
            self.branch2 = nn.Sequential(
                nn.Conv3d(numChannels, int(numChannels/2), kernel_size=3, stride=1,padding=1),
                nn.AvgPool3d(3, stride=2, padding=1)
            )

            self.branch3 = nn.Sequential(
                nn.Conv3d(numChannels, int(numChannels/2), kernel_size=3, stride=1,padding=1),
                nn.MaxPool3d(3, stride=2, padding=1)
            )
        elif (not elu) and batch_norm:
            self.branch0 = nn.Sequential(nn.Conv3d(numChannels, int(numChannels/2), kernel_size=3, stride=2,padding=1),
                                         nn.BatchNorm3d(int(numChannels/2),eps=0.001))
            self.branch1 = nn.Sequential(nn.Conv3d(numChannels, int(numChannels/2), kernel_size=1, stride=2),
                                         nn.BatchNorm3d(int(numChannels/2),eps=0.001))
            self.branch2 = nn.Sequential(
                nn.Conv3d(numChannels, int(numChannels/2), kernel_size=3, stride=1,padding=1),
                nn.AvgPool3d(3, stride=2, padding=1),
                nn.BatchNorm3d(int(numChannels/2),eps=0.001)
            )

            self.branch3 = nn.Sequential(
                nn.Conv3d(numChannels, int(numChannels/2), kernel_size=3, stride=1,padding=1),
                nn.MaxPool3d(3, stride=2, padding=1),
                nn.BatchNorm3d(int(numChannels/2),eps=0.001)
            )
        else:
            self.branch0 = BasicConv3d(numChannels, int(numChannels/2), kernel_size=3, stride=2,padding=1)
            self.branch1 = BasicConv3d(numChannels, int(numChannels/2), kernel_size=1, stride=2)
            self.branch2 = nn.Sequential(
                nn.Conv3d(numChannels, int(numChannels/2), kernel_size=3, stride=1,padding=1),
                nn.AvgPool3d(3, stride=2, padding=1),
                nn.BatchNorm3d(int(numChannels/2),eps=0.001),nn.ELU()
            )

            self.branch3 = nn.Sequential(
                nn.Conv3d(numChannels, int(numChannels/2), kernel_size=3, stride=1,padding=1),
                nn.MaxPool3d(3, stride=2, padding=1),
                nn.BatchNorm3d(int(numChannels/2),eps=0.001),nn.ELU()
            )



    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

    
class Voxception_Resnet_Layer(nn.Module):

    def __init__(self,numChannels):
        super(Voxception_Resnet_Layer, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv3d(numChannels, int(numChannels/4), kernel_size=3, stride=1,padding=1),
            nn.Conv3d(int(numChannels/4), int(numChannels/2), kernel_size=3, stride=1,padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv3d(numChannels, int(numChannels/4), kernel_size=1, stride=1),
            BasicConv3d(int(numChannels/4), int(numChannels/4), kernel_size=3, stride=1,padding=1),
            nn.Conv3d(int(numChannels/4), int(numChannels/2), kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = x
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x1, x2), 1)
        return out+x0


class BasicConv3d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding) # verify bias false
        self.bn = nn.BatchNorm3d(out_planes,eps=0.001)
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.elu(x)
        return x

    
class Resdrop(nn.Module):
    def __init__(self,drop_layer,p):
        super(Resdrop, self).__init__()
        self.nn_module = drop_layer
        self.death_rate = p

    
    def forward(self, x):
        residual = x
        sample=0
        if not self.training or torch.rand(1)[0] >= self.death_rate:
            residual = self.nn_module(residual)
            if self.training:
                residual /= (1. - self.death_rate)
            x = x + residual
        return x



class VoxceptionNet(nn.Module):
    def __init__(self,n_classes):
        super(VoxceptionNet, self).__init__() 
        
        self.input_conv = nn.Conv3d(1, 32, kernel_size=3, stride=1)
        self.b1 = nn.Sequential(Resdrop(nn.Sequential(nn.BatchNorm3d(32,eps=0.001),nn.ELU(),Voxception_Resnet_Layer(numChannels=32)),0.95),
                                Resdrop(nn.Sequential(nn.BatchNorm3d(32,eps=0.001),nn.ELU(),Voxception_Resnet_Layer(numChannels=32)),0.90),
                                Resdrop(nn.Sequential(nn.BatchNorm3d(32,eps=0.001),nn.ELU(),Voxception_Resnet_Layer(numChannels=32)),0.80),
                                nn.BatchNorm3d(32,eps=0.001),nn.ELU(),
                                Voxception_Downsample_Layer(numChannels=32,batch_norm=True))
    
        self.b2 = nn.Sequential(Resdrop(nn.Sequential(nn.BatchNorm3d(64,eps=0.001),nn.ELU(),Voxception_Resnet_Layer(numChannels=64)),0.7),
                                Resdrop(nn.Sequential(nn.BatchNorm3d(64,eps=0.001),nn.ELU(),Voxception_Resnet_Layer(numChannels=64)),0.6),
                                Resdrop(nn.Sequential(nn.BatchNorm3d(64,eps=0.001),nn.ELU(),Voxception_Resnet_Layer(numChannels=64)),0.5),
                                nn.BatchNorm3d(64,eps=0.001),nn.ELU(),
                                Voxception_Downsample_Layer(numChannels=64,batch_norm=True))
        
        self.b3 = nn.Sequential(Resdrop(nn.Sequential(nn.BatchNorm3d(128,eps=0.001),nn.ELU(),Voxception_Resnet_Layer(numChannels=128)),0.5),
                                Resdrop(nn.Sequential(nn.BatchNorm3d(128,eps=0.001),nn.ELU(),Voxception_Resnet_Layer(numChannels=128)),0.45),
                                Resdrop(nn.Sequential(nn.BatchNorm3d(128,eps=0.001),nn.ELU(),Voxception_Resnet_Layer(numChannels=128)),0.40),
                                nn.BatchNorm3d(128,eps=0.001),nn.ELU(),
                                Voxception_Downsample_Layer(numChannels=128))
        
        self.b4 = nn.Sequential(Resdrop(nn.Sequential(nn.BatchNorm3d(256,eps=0.001),nn.ELU(),Voxception_Resnet_Layer(numChannels=256)),0.35),
                                Resdrop(nn.Sequential(nn.BatchNorm3d(256,eps=0.001),nn.ELU(),Voxception_Resnet_Layer(numChannels=256)),0.30),
                                Resdrop(nn.Sequential(nn.BatchNorm3d(256,eps=0.001),nn.ELU(),Voxception_Resnet_Layer(numChannels=256)),0.25),
                                nn.BatchNorm3d(256,eps=0.001),nn.ELU(),
                                Voxception_Downsample_Layer(numChannels=256,elu=True))

        self.b5 = nn.Sequential(Resdrop(nn.Sequential(nn.Conv3d(512,512,3,1,1),nn.BatchNorm3d(512,eps=0.001)),0.5),nn.ELU())
        #add view numFeatures
        self.fc = nn.Sequential(nn.Linear(512,512),nn.BatchNorm1d(512),nn.ELU(),nn.Linear(512,n_classes),nn.Softmax(1))


                              
    def forward(self,x):
        x = self.input_conv(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        out = F.max_pool3d(x, kernel_size=x.size()[2:])
        out = out.view(-1,512)
        output = self.fc(out)
        return output
        
