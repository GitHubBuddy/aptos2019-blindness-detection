import os
import numpy as np
import pandas as pd
from .transform import *
import pdb

#import math
import cv2
#import PIL.Image as Image
#import matplotlib.pyplot as plt
import random
#from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler



class BlindnessDataset(Dataset):
    def __init__(self, datapath, mode='train', transform=None):
        super(BlindnessDataset, self).__init__()
        self.path = datapath
        self.mode = mode
        self.transform = transform
#        self.num_classes = 5
        #load the excel file:
        if mode=='train':
            images = pd.read_csv(self.path + '/train.csv')
#            images = pd.read_csv('train.csv')
            filenames = images['id_code'].tolist()
            labels = images['diagnosis'].tolist()
        else:
            images = pd.read_csv(self.path + '/test.csv')
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
#        img = Image.open('./data/{}/{}.png'.format(self.mode, filename))
        img = cv2.imread(self.path + '/{}_processed/{}.png'.format(self.mode, filename))
#        print(img.shape)
        if self.transform: img = self.transform(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
###define our own transform:
#        img = circle_crop(img)
#        height = width = 224
#        img = cv2.resize(img, (height, width))
        if random.randint(0,1)==1:
            img = cv2.flip(img, 0)
        angle = random.randint(0,359)
        img = rotate(img, angle, center=None, scale=1.0)
        #Rotation first and then GaussianBlur
        img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img , (0,0) , sigmaX=10) ,-4 ,128)
###Convert to tensor:
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        img = img.float().div(255)
#        img = img.float().div(255).unsqueeze(0)
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
                                                num_workers=16, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, \
                                                num_workers=16, sampler=valid_sampler)
    
    return train_loader, validation_loader





if __name__ == '__main__':
    mydata = BlindnessDataset('train')
    train_loader, validation_loader = train_valid_dataset(mydata, 4, validation_ratio=0.2, shuffle=True)
    i = 0
    for d in train_loader:
        if i>4: break
        i += 1
        print(len(d))
