#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 23:06:09 2021

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
sys.path.insert(0,'/content/drive/MyDrive/Colab Notebooks/DiCOVA_Challenge_Drive')
from configs import *

class dataset_utils(object):
    def __init__(self):
        ''' '''
    def getaudio(self, ID):
        ''' gets audio signal given subject ID '''
        dir_path = configs['dataset']['audio_path']
        audio_path = dir_path + '/' + ID + '.flac'
        audio = self.read_audio(audio_path, configs['dataset']['sampling_rate'])
        return audio
        
    def read_audio(self, file_path,sampling_rate):
        ''' reads audio file and normalizes amplitudes'''
        fs=librosa.get_samplerate(file_path)
        try:
          s,_ = librosa.load(file_path,sr=sampling_rate)
          if np.mean(s)==0 or len(s)<1024:
            raise ValueError()
          # waveform level amplitude normalization
          s = s/np.max(np.abs(s))
        except ValueError:
          s = None
          print("Read audio failed for "+file_path)		
        return s

        return X,y
    
    def get_label(self, ID):
        ''' Gets labels given ID'''
        metadata_path = configs['dataset']['metadata_path']
        metadata = pd.read_csv(metadata_path)
        covid_status = metadata.loc[metadata['File_name'] == ID, 'Covid_status'].iloc[0]
        if covid_status == 'p':
            label = 1
        elif covid_status == 'n':
            label = 0
        return label
        
    def reshape_image(self, input_image, n_rows, n_cols):
        ''' stitch images together if smaller than configs, resize if larger'''
 
        img_rows, img_cols = np.shape(input_image)
        if img_cols > n_cols:
            resized_image = cv2.resize(input_image, (n_cols, n_rows))
          
        else:
            multiples = n_cols // img_cols 
            left_over = n_cols % img_cols
            cropped_image = input_image[:,:left_over]
            resized_image = np.hstack([input_image]*multiples)
            resized_image = np.hstack((resized_image,cropped_image))
        
        return resized_image
            
    def stack_images_copies(self, input_image):
        input_image = np.expand_dims(input_image,0)
        stacked_image = np.concatenate([input_image]*3,0)
        return stacked_image
    
    def normalize_images(self, image):
        image = image/np.max(image)
        return image

class Mel_log_spect(object):
    def __init__(self):
        self.nfft = configs['mel_log_spect']['fftl']
        self.num_mels = configs['mel_log_spect']['num_mels']
        self.hop_length = configs['mel_log_spect']['hop_length']
        self.top_db = configs['mel_log_spect']['top_db']
        self.sr = configs['mel_log_spect']['sr']

    def feature_normalize(self, x):
        log_min = np.min(x)
        x = x - log_min
        x = x / self.top_db
        x = x.T
        return x

    def get_Mel_log_spect(self, y):
        y = librosa.util.normalize(S=y)
        spect = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=self.nfft,
                                               hop_length=self.hop_length, n_mels=self.num_mels)
        log_spect = librosa.core.amplitude_to_db(spect, ref=1.0, top_db=self.top_db)
        log_spect = self.feature_normalize(log_spect)
        return log_spect

    def norm_Mel_log_spect_to_amplitude(self, feature):
        feature = feature * self.top_db
        spect = librosa.core.db_to_amplitude(feature, ref=1.0)
        return spect

    def audio_from_spect(self, feature):
        spect = self.norm_Mel_log_spect_to_amplitude(feature)
        audio = librosa.feature.inverse.mel_to_audio(spect.T, sr=self.sr, n_fft=self.nfft, hop_length=self.hop_length)
        return audio

    def convert_and_write(self, load_path, write_path):
        y, sr = librosa.core.load(path=load_path, sr=self.sr)
        feature = self.get_Mel_log_spect(y, n_mels=self.num_mels)
        audio = self.audio_from_spect(feature)
        librosa.output.write_wav(write_path, y=audio, sr=self.sr, norm=True)


def read_directory(dir_path):
    all_files = os.listdir(dir_path)
    subjects = []
    for file in all_files:
        subjects.append(file.split('.')[0])
    return subjects

def plot_mfcc(mfcc):
    fig, ax = plt.subplots()
    cax = ax.imshow(mfcc, vmin = 0, vmax = 1,interpolation='nearest', cmap=cm.hot, origin='lower')
    ax.set_title('MFCC')
    plt.show()

def get_mfcc_sizes(data):
    mfcc_size = []
    for i in range(len(data)):
        mfcc, y = data[i]
        mfcc_size.append(np.shape(mfcc)[1])
    return mfcc_size

def get_fold(txt_path):
    fold = []
    with open(txt_path) as fp:
        for line in fp:
            ID = line.split()[0]
            fold.append(ID) 
    return fold 


def scoring(refs, sys_outs):
    """
        inputs::
        refs: a txt file with a list of labels for each wav-fileid in the format: <id> <label>
        sys_outs: a txt file with a list of scores (probability of being covid positive) for each wav-fileid in the format: <id> <score>
        threshold (optional): a np.array(), like np.arrange(0,1,.01), sweeping for AUC
        outputs::
        """
    reference_labels = refs
    sys_scores = sys_outs
    thresholds = np.arange(0, 1, 0.01)
    # # Read the ground truth labels into a dictionary
    # data = open(refs).readlines()
    # reference_labels = {}
    # categories = ['n', 'p']
    # for line in data:
    #     key, val = line.strip().split()
    #     reference_labels[key] = categories.index(val)

    # # Read the system scores into a dictionary
    # data = open(sys_outs).readlines()
    # sys_scores = {}
    # for line in data:
    #     key, val = line.strip().split()
    #     sys_scores[key] = float(val)
    # del data

    # Ensure all files in the reference have system scores and vice-versa
    if len(sys_scores) != len(reference_labels):
        print("Expected the score file to have scores for all files in reference and no duplicates/extra entries")
        return None
    # %%

    # Arrays to store true positives, false positives, true negatives, false negatives
    TP = np.zeros((len(reference_labels), len(thresholds)))
    TN = np.zeros((len(reference_labels), len(thresholds)))
    keyCnt = -1
    for key in sys_scores:  # Repeat for each recording
        keyCnt += 1
        sys_labels = (sys_scores[key] >= thresholds) * 1  # System label for a range of thresholds as binary 0/1
        gt = reference_labels[key]

        ind = np.where(sys_labels == gt)  # system label matches the ground truth
        if gt == 1:  # ground-truth label=1: True positives
            TP[keyCnt, ind] = 1
        else:  # ground-truth label=0: True negatives
            TN[keyCnt, ind] = 1

    total_positives = sum(reference_labels.values())  # Total number of positive samples
    total_negatives = len(reference_labels) - total_positives  # Total number of negative samples

    TP = np.sum(TP, axis=0)  # Sum across the recordings
    TN = np.sum(TN, axis=0)

    TPR = TP / total_positives  # True positive rate: #true_positives/#total_positives
    TNR = TN / total_negatives  # True negative rate: #true_negatives/#total_negatives

    AUC = auc(1 - TNR, TPR)  # AUC

    ind = np.where(TPR >= 0.8)[0]
    sensitivity = TPR[ind[-1]]
    specificity = TNR[ind[-1]]

    # pack the performance metrics in a dictionary to save & return
    # Each performance metric (except AUC) is a array for different threshold values
    # Specificity at 90% sensitivity
    scores = {'TPR': TPR,
              'FPR': 1 - TNR,
              'AUC': AUC,
              'sensitivity': sensitivity,
              'specificity': specificity,
              'thresholds': thresholds}

    # with open(out_file, "wb") as f:
    #     pickle.dump(scores, f)
    return scores

def summary(folname, scores, iterations):
    # folname = sys.argv[1]
    num_files = 1
    R = []
    for i in range(num_files):
        # res = pickle.load(open(folname + "/fold_{}/val_results.pkl".format(i + 1), 'rb'))
        # res = pickle.load(open(scores))
        res = joblib.load(scores)
        R.append(res)

    # Plot ROC curves
    clr_1 = 'tab:green'
    clr_2 = 'tab:green'
    clr_3 = 'k'
    data_x, data_y, data_auc = [], [], []
    for i in range(num_files):
        data_x.append(R[i]['FPR'].tolist())
        data_y.append(R[i]['TPR'].tolist())
        data_auc.append(R[i]['AUC'] * 100)
        plt.plot(data_x[i], data_y[i], label='V-' + str(i + 1) + ', auc=' + str(np.round(data_auc[i], 2)), c=clr_1,
                 alpha=0.2)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    plt.plot(np.mean(data_x, axis=0), np.mean(data_y, axis=0),
             label='AVG, auc=' + str(np.round(np.mean(np.array(data_auc)), 2)), c=clr_2, alpha=1, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', label='chance', c=clr_3, alpha=.5)
    plt.legend(loc='lower right', frameon=False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.grid(color='gray', linestyle='--', linewidth=1, alpha=.3)
    plt.text(0, 1, 'PATIENT-LEVEL ROC', color='gray', fontsize=12)

    plt.gca().set_xlabel('FALSE POSITIVE RATE')
    plt.gca().set_ylabel('TRUE POSITIVE RATE')
    plt.savefig(os.path.join(folname, 'val_roc_plot_' + str(iterations) + '.pdf'), bbox_inches='tight')
    plt.close()

    sensitivities = [R[i]['sensitivity'] * 100 for i in range(num_files)]
    specificities = [R[i]['specificity'] * 100 for i in range(num_files)]

    with open(os.path.join(folname, 'val_summary_metrics.txt'), 'w') as f:
        f.write("Sensitivities: " + " ".join([str(round(item, 2)) for item in sensitivities]) + "\n")
        f.write("Specificities: " + " ".join([str(round(item, 2)) for item in specificities]) + "\n")
        f.write("AUCs: " + " ".join([str(round(item, 2)) for item in data_auc]) + "\n")
        f.write(
            "Average sensitivity: " + str(np.round(np.mean(np.array(sensitivities)), 2)) + " standard deviation:" + str(
                np.round(np.std(np.array(sensitivities)), 2)) + "\n")
        f.write(
            "Average specificity: " + str(np.round(np.mean(np.array(specificities)), 2)) + " standard deviation:" + str(
                np.round(np.std(np.array(specificities)), 2)) + "\n")
        f.write("Average AUC: " + str(np.round(np.mean(np.array(data_auc)), 2)) + " standard deviation:" + str(
            np.round(np.std(np.array(data_auc)), 2)) + "\n")
    return np.round(np.mean(np.array(data_auc)), 2)



def summary_1fold(scores, iterations):
    # folname = sys.argv[1]
    num_files = 1
    R = scores
    # for i in range(num_files):
    #     # res = pickle.load(open(folname + "/fold_{}/val_results.pkl".format(i + 1), 'rb'))
    #     # res = pickle.load(open(scores))
    #     res = joblib.load(scores)
    #     R.append(res)
    folname = 'hi'

    # Plot ROC curves
    clr_1 = 'tab:green'
    clr_2 = 'tab:green'
    clr_3 = 'k'
    data_x, data_y, data_auc = [], [], []
    # for i in range(num_files):
    data_x.append(R['FPR'].tolist())
    data_y.append(R['TPR'].tolist())
    data_auc.append(R['AUC'] * 100)
    plt.plot(data_x, data_y, label='V-' + ', auc=' + str(np.round(data_auc, 2)), c=clr_1,
              alpha=0.2)
    
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    plt.plot(np.mean(data_x, axis=0), np.mean(data_y, axis=0),
             label='AVG, auc=' + str(np.round(np.mean(np.array(data_auc)), 2)), c=clr_2, alpha=1, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', label='chance', c=clr_3, alpha=.5)
    plt.legend(loc='lower right', frameon=False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.grid(color='gray', linestyle='--', linewidth=1, alpha=.3)
    plt.text(0, 1, 'PATIENT-LEVEL ROC', color='gray', fontsize=12)

    plt.gca().set_xlabel('FALSE POSITIVE RATE')
    plt.gca().set_ylabel('TRUE POSITIVE RATE')
    plt.savefig(os.path.join(folname, 'val_roc_plot_' + str(iterations) + '.pdf'), bbox_inches='tight')
    plt.close()

    sensitivities = [R[i]['sensitivity'] * 100 for i in range(num_files)]
    specificities = [R[i]['specificity'] * 100 for i in range(num_files)]

    with open(os.path.join(folname, 'val_summary_metrics.txt'), 'w') as f:
        f.write("Sensitivities: " + " ".join([str(round(item, 2)) for item in sensitivities]) + "\n")
        f.write("Specificities: " + " ".join([str(round(item, 2)) for item in specificities]) + "\n")
        f.write("AUCs: " + " ".join([str(round(item, 2)) for item in data_auc]) + "\n")
        f.write(
            "Average sensitivity: " + str(np.round(np.mean(np.array(sensitivities)), 2)) + " standard deviation:" + str(
                np.round(np.std(np.array(sensitivities)), 2)) + "\n")
        f.write(
            "Average specificity: " + str(np.round(np.mean(np.array(specificities)), 2)) + " standard deviation:" + str(
                np.round(np.std(np.array(specificities)), 2)) + "\n")
        f.write("Average AUC: " + str(np.round(np.mean(np.array(data_auc)), 2)) + " standard deviation:" + str(
            np.round(np.std(np.array(data_auc)), 2)) + "\n")
    return np.round(np.mean(np.array(data_auc)), 2)


def feed_to_scoring(IDs, labels, predictions):
    # IDs = all_IDs[epoch]
    # labels = all_labels[epoch]
    # predictions = all_predictions[epoch]

    reference_labels = {}
    sys_scores = {}
    for i in range(len(IDs)):
        reference_labels[IDs[i]] = int(labels[i])
        sys_scores[IDs[i]] = predictions[i]
    return reference_labels, sys_scores
    
    
def get_cm(labels, predictions, threshold):
    num_samples = len(labels)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(num_samples):
        if (labels[i] == 1) and (predictions[i] > threshold):
            TP += 1
        if (labels[i] == 0) and (predictions[i] > threshold):
            FP += 1
        if (labels[i] == 1) and (predictions[i] < threshold):  
            FN += 1
        if (labels[i] == 0) and (predictions[i] < threshold): 
            TN += 1
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    return TP, TN, FP, FN, accuracy