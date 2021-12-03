import numpy as np
import pickle
import collections
import scipy.io
import time
import datetime

from sklearn.metrics import *
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

from predict import *



#
#           Main of the Module
#


if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_model = torch.load('Models/cnn2d_small_cubes_16_[2020-11-04__3-35].pt')
    best_model = best_model.to(device)
    best_model.eval()


    loader_ = torch.load('Data_Loaders/train_loader_16.pt')

    a = []
    b = []
    all_images = []
    all_labels = []

    for images, labels in loader_:
        for i in images:
            a.append(i)
        for j in labels:
            b.append(j) 
    all_images = a
    all_labels = b
    nb_total = len(all_images)

    predictions = predict_classes(best_model, all_images)
    
    acc = accuracy_score(all_labels, predictions)
    conf_mat = confusion_matrix(all_labels, predictions)

    print('\n')
    print(conf_mat)
    print('\n')
    print('Total Accuracy:  ' + str(acc))
    print('\n')
    
    print('Accuracy Class 1:  ' + str(conf_mat[0, 0] / nb_total))
    print('Accuracy Class 2:  ' + str(conf_mat[1, 1] / nb_total))
    print('Accuracy Class 3:  ' + str(conf_mat[2, 2] / nb_total))


