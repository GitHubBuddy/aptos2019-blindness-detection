import os
import numpy as np
import pandas as pd
from .transform import *

#import math
#import cv2
import PIL.Image as Image
#import matplotlib.pyplot as plt
import random
#from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler



class BlindnessDataset(Dataset):
    def __init__(self, mode='train', transform=None):
        super(BlindnessDataset, self).__init__()
#        self.path = datapath
        self.mode = mode
        self.transform = transform
        self.num_classes = 5
        #load the excel file:
        if mode=='train':
            images = pd.read_csv('./data/train.csv')
#            images = pd.read_csv('train.csv')
            filenames = images['id_code'].tolist()
            labels = images['diagnosis'].tolist()
        else:
            images = pd.read_csv('./data/test.csv')
#            images = pd.read_csv('test.csv')
            filenames = images['id_code'].tolist()
            labels = None
        
        self.filenames = filenames
        self.labels = labels
    

    
    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, index):
        if self.labels:
            label = self.labels[index]
        else: label = None
        filename = self.filenames[index]
        img = Image.open('./data/{}/{}.png'.format(self.mode, filename))
        if self.transform: img = self.transform(img)
#        img = cv2.imread('./{}/{}.png'.format(self.mode, filename))
#        img = cv2.resize(img, (height, width))
        return img, label







def train_valid_dataset(dataset, batch_size, validation_ratio=0.2, shuffle=True):
    #random_seed= 42

    #Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_ratio * dataset_size))
    if shuffle:
    #    np.random.seed(random_seed)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, \
                                                num_workers=0, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, \
                                                num_workers=0, sampler=valid_sampler)
    
    return train_loader, validation_loader





if __name__ == '__main__':
    mydata = BlindnessDataset('train')
    train_loader, validation_loader = train_valid_dataset(mydata, 4, validation_ratio=0.2, shuffle=True)
    i = 0
    for d in train_loader:
        if i>4: break
        i += 1
        print(len(d))