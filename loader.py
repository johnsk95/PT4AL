import glob
import os
from PIL import Image, ImageFilter

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import cv2

random.seed(0)
np.random.seed(0)

class RotationLoader(Dataset):
    def __init__(self, is_train=0, transform=None, path='/workspace/ACTIVE/DATA'):
        self.is_train = is_train
        self.transform = transform
        self.h_flip = transforms.RandomHorizontalFlip(p=1)
        if self.is_train == 0: # train
            self.img_path = glob.glob('./DATA3/train/*/*')
        else:
            self.img_path = glob.glob('./DATA3/train/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        label = int(self.img_path[idx].split('/')[-2])

        if self.is_train ==0:
            img = self.transform(img)
            rotation = np.random.randint(4)
            img = torch.rot90(img, rotation, [1,2])
       
            return img, rotation
        else:
            img = self.transform(img)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            rotation = 0
            return img, img1, img2, img3, 0,1,2,3, self.img_path[idx]

class Loader2(Dataset):
    def __init__(self, is_train=0, transform=None, path='/workspace/ACTIVE/DATA', path_list=None):
        self.is_train = is_train
        self.transform = transform
        self.path_list = path_list

        if self.is_train == 0: # train
            self.img_path = path_list
        else:
            if path_list is None:
                self.img_path = glob.glob('./DATA3/test/*/*')
            else:
                self.img_path = path_list
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if self.is_train == 0:
            img = cv2.imread(self.img_path[idx][:-1])
        else:
            if self.path_list is None:
                img = cv2.imread(self.img_path[idx])
            else:
                img = cv2.imread(self.img_path[idx][:-1])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = int(self.img_path[idx].split('/')[-2])

        return img, label

class Loader(Dataset):
    def __init__(self, is_train=0, transform=None, path='/workspace/ACTIVE/DATA'):
        self.is_train = is_train
        self.transform = transform
        if self.is_train == 0: # train
            self.img_path = glob.glob('./DATA3/train/*/*')
        else:
            self.img_path = glob.glob('./DATA3/test/*/*')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        img = self.transform(img)
        label = int(self.img_path[idx].split('/')[-2])

        return img, label
