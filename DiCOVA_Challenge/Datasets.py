#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 23:25:51 2021

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
import espnet.transform.spec_augment as SPEC
import sys
sys.path.insert(0,'/content/drive/MyDrive/Colab Notebooks/DiCOVA_Challenge_Drive')
from configs import *
import utils


class Dataset_mfcc(object):
    def __init__(self, subject_IDs):
        'Initialization'
        self.subject_IDs = subject_IDs
        self.dataset_utils = utils.dataset_utils()
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.subject_IDs)

    def __getitem__(self, index):
        'Generates one sample of the data'
        #Select sample
        ID = self.subject_IDs[index]
        
        #X = self.getaudio(ID) #Uncomment if input data is raw audio signal
        X = self.getmfcc(ID)
        X = self.dataset_utils.reshape_image(X)
        X = self.dataset_utils.reshape_image(X, configs['dataset']['mfcc_resized_n_rows'], configs['dataset']['mfcc_resized_n_cols'])
        X = self.dataset_utils.normalize_images(X)
        X = self.dataset_utils.stack_images_copies(X)
        y = self.dataset_utils.get_label(ID)
        return X, y, ID


    def getmfcc(self, ID):
        ''' get mfcc given subject ID ''' 
        dir_path = configs['dataset']['mfcc_path']
        mfcc_path = dir_path + '/' + ID + '_mfcc.pkl'
        f = open(mfcc_path, 'rb')   # 'r' for reading; can be omitted
        mfcc = pickle.load(f)         # load file content as mydict
        f.close()  

        return mfcc

class Dataset_mel_log_spect(object):
    def __init__(self, subject_IDs, transform):
        'Initialization'
        self.subject_IDs = subject_IDs
        self.dataset_utils = utils.dataset_utils()
        self.mel_log_spect = utils.Mel_log_spect()
        self.transform = transform
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.subject_IDs)

    def __getitem__(self, index):
        'Generates one sample of the data'
        #Select sample
        ID = self.subject_IDs[index]
        
        X = self.dataset_utils.getaudio(ID) #Uncomment if input data is raw audio signal
        X = self.mel_log_spect.get_Mel_log_spect(X)
        
        if self.transform:
            time_width = round(X.shape[0]*0.1)
            X = aug_feats = SPEC.spec_augment(X, resize_mode='PIL', max_time_warp=80,
                                                                   max_freq_width=20, n_freq_mask=1,
                                                                   max_time_width=time_width, n_time_mask=2,
                                                                   inplace=False, replace_with_zero=True)
        
        X = np.rot90(X)
        X = self.dataset_utils.reshape_image(X, configs['dataset']['mls_resized_n_rows'], configs['dataset']['mls_resized_n_cols'])
        X = self.dataset_utils.stack_images_copies(X)
        y = self.dataset_utils.get_label(ID)
        return X, y, ID

    def getlabels(self):
        'gets labels'
        labels = np.zeros(len(self.subject_IDs))

        for idx in range(len(self.subject_IDs)):
            ID = self.subject_IDs[idx]
            y = self.dataset_utils.get_label(ID)
            labels[idx] = y
            
        return labels
        