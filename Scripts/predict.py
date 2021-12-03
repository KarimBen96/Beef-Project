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
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid




def predict_class(model, image):
    """
    """
    image = image.to(torch.device('cuda'))
    with torch.no_grad():
        model.eval()
        output, _ = model(image.unsqueeze_(0))
    softmax = torch.exp(output).cpu()
    prob = list(softmax.detach().numpy())
    prediction = np.argmax(prob, axis=1)

    return prediction


def predict_classes(model, images):
    """
    - images: a list of images and NOT a data loader
    """
    predictions = []
    for img in images:
        predictions.append(predict_class(model, img))
    return predictions    