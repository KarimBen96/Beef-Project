import numpy as np
import scipy.io

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid





def get_flatten_image(model, image):
    """
    """
    image = image.to(torch.device('cuda'))
    with torch.no_grad():
        model.eval()
        _, flatten_image = model(image.unsqueeze_(0))

    return flatten_image






#
#           Main of the module
#



if __name__ == '__main__':
    
    loader_ = torch.load('Data_Loaders/all_images_train_loader.pt')

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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_model = torch.load('Models/cnn3d_[2020-10-30__14-15].pt')
    best_model = best_model.to(device)
    best_model.eval()

        
    flatten_images = []
    for i in all_images:
        flatten_images.append(np.array(get_flatten_image(best_model, i).cpu()))
        
    flatten_images_save = np.array(flatten_images).reshape(186, 3136)
    all_labels_save = np.array(all_labels).transpose() 
    
    scipy.io.savemat('Flattened_Images/beef_flatten_images_3136.mat', {'X': flatten_images_save, 'Y': all_labels_save})