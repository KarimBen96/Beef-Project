import numpy as np
import pickle
import collections
import scipy.io
import time
import datetime

import sklearn.metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.optim import Adam, SGD



def load_data(images_file, labels_file):
    """
    """
    X = np.load(images_file)
    y = np.load(labels_file)
    
    return X, y



def load_adundance_maps(images_file, labels_file):
    """
    """
    
    mat_1 = scipy.io.loadmat(images_file)
    all_abundance_maps = (mat_1['Ab_Maps_Meat'])
    
    all_labels = np.load(labels_file)
    
    X = all_abundance_maps
    y = all_labels[:, 1]
    y = y - 1
    
    return X, y



def reshape_abandance_maps_4d(X, new_shape):
    """
    """
    a, b, c, d = new_shape
    X = X.reshape(a)
    
    new_X = np.empty(new_shape)
    for i in range(a):
        new_X[i] = X[i].reshape(b, c, d)
        
    return new_X



def reshape_abandance_maps_5d(X, new_shape):
    """
    """
    a, b, c, d, e = new_shape
    X = X.reshape(a)
    
    new_X = np.empty(new_shape)
    for i in range(a):
        new_X[i] = X[i].reshape(b, c, d, e)
        
    return new_X



def data_normalization(X):
    """
    """
    a = np.mean(X)
    b = np.std(X)
    X = (X - a) / b
    
    return X



def train_test_split_(X, y, seed_):
    """
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.19, random_state=seed_, shuffle=True)
    return X_train, X_test, y_train, y_test
    



#
#           Main of the Module
#


if __name__ == '__main__':
    
    X, y = load_adundance_maps('Data_Raw/Abundance_Maps/Ab_Maps_Beef_128.mat', 
                     'Data_Raw/all_labels.npy')
    
    X = reshape_abandance_maps_4d(X, (186, 3, 128, 128))
    X = data_normalization(X)

    print(X.shape)

    np.save('Data_Pre_Processed/X_128_preprocessed', X)
    np.save('Data_Pre_Processed/y_128_preprocessed', y)

    
    
    