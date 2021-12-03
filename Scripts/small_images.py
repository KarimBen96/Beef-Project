import numpy as np
import pickle
import collections
import scipy.io
import time
import datetime
import copy

import sklearn.metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid


from CNN_3D_Model import *
from CNN_2D_Model import *
from predict import *




def reduce_cube(big_cube, new_l, new_w):
    """
    - big_cube is (19, 700, 575)
    """
    
    a, b, c = big_cube.shape
    m = int((b * c) / (new_l * new_w))
    
    small_cubes = big_cube.reshape(m, a, new_l, new_w)
    
    return small_cubes



#
#           Main of the Module
#


if __name__ == '__main__':
    
    # Loading Data
    X = np.load('Data_Pre_Processed\X_128_preprocessed.npy')
    y = np.load('Data_Pre_Processed\y_128_preprocessed.npy')
    
    X = X.reshape(X.shape[0], 3, 128, 128)
    
    all_small_cubes = []
    final_small_cubes = []
    final_y = []
    
    for img in X:
        all_small_cubes.append(reduce_cube(img, 16, 16))
        
    for img in all_small_cubes:
        for i in img:
            final_small_cubes.append(i)
        
    for i in range(len(y)):    
        for j in range(int(len(final_small_cubes) / len(y))):
            final_y.append(y[i])


    np.save('Data_Pre_Processed\X_16_preprocessed.npy', np.array(final_small_cubes))
    np.save('Data_Pre_Processed\y_16_preprocessed.npy', np.array(final_y))


    print(np.array(final_small_cubes).shape)
    
    