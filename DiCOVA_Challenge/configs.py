#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 19:01:16 2021

@author: yash_wani
"""

configs = {}

mfcc = {'n_mfcc' : 39,
        'n_mels' : 64,
        'fmax' : 22050,
        'add_deltas' : True,
        'add_delta_deltas' : True,
        'test': 42
}

mel_log_spect = {'fftl' : 1024 ,
                'num_mels': 80,
                'hop_length': 160,
                'top_db': 120,
                'sr': 16000
    
}

dataset = {'audio_path': '/content/drive/MyDrive/Research_Data/DICOVA_data_baseline/DiCOVA_Train_Val_Data_Release/AUDIO',
           'sampling_rate': 44100,
           'mfcc_path': '/content/drive/MyDrive/Research_Data/DICOVA_data_baseline/DiCOVA_baseline/feats',
           'metadata_path': '/content/drive/MyDrive/Research_Data/DICOVA_data_baseline/DiCOVA_Train_Val_Data_Release/metadata.csv',
           'mfcc_resized_n_rows': 117,
           'mfcc_resized_n_cols': 225,
           'mls_resized_n_rows': 1000,
           'mls_resized_n_cols': 80,
           'fold_path' : '/content/drive/MyDrive/Research_Data/DICOVA_data_baseline/DiCOVA_Train_Val_Data_Release/LISTS',
           'train_path_1': '/content/drive/MyDrive/Research_Data/DICOVA_data_baseline/DiCOVA_Train_Val_Data_Release/LISTS/train_fold_1.txt',
           'val_path_1' : '/content/drive/MyDrive/Research_Data/DICOVA_data_baseline/DiCOVA_Train_Val_Data_Release/LISTS/val_fold_1.txt'
}

resnet18_hp = {'batch_size' : 64,
                'batch_size_test' : 5,
                'lr' : 0.0001,
                'num_epochs' : 150,
                'num_epochs_test':10,
                'spec_augment': True
    
}

configs['mfcc'] = mfcc
configs['dataset'] = dataset
configs['resnet18_hp'] = resnet18_hp
configs['mel_log_spect'] = mel_log_spect