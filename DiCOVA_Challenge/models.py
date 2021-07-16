#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 00:08:54 2021

@author: yash_wani
"""
import os
import librosa
import time
import matplotlib.pyplot as plt
import torch
from sklearn.utils import shuffle
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import auc
import torch.optim as optim
import torch.nn.functional as functional
import joblib
import cv2
import pandas as pd
import pickle
import torchvision.models as models
from matplotlib import cm
import sys

#import local files
sys.path.insert(0,'/content/drive/MyDrive/Colab Notebooks/DiCOVA_Challenge_Drive')
from configs import *
import utils
from Datasets import Dataset_mfcc

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained = True)
        #freeze above layers
        # for param in self.model.parameters():
        #     param.requires_grad = False
        #add few final linear layers
        self.model.fc = nn.Sequential(nn.Linear(in_features = 512, out_features = 128, bias = True),
                            nn.ReLU(inplace = True),
                            nn.Linear(in_features = 128, out_features = 2, bias = True),
                            nn.Softmax())
        

    def forward(self, x):
        x = self.model(x)
        return x 
                       
                       
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained = True)
        #freeze above layers
        for param in self.model.parameters():
            param.requires_grad = False
        #add few final linear layers
        self.model.fc = nn.Sequential(nn.Linear(in_features = 512, out_features = 128, bias = True),
                            nn.ReLU(inplace = True),
                            nn.Linear(in_features = 128, out_features = 2, bias = True),
                            nn.Softmax())
        

    def forward(self, x):
        x = self.model(x)
        return x 