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




def train_model(model_untrained, train_loader, optimizer, loss_function, nb_epochs=1, 
                validation_loader=None, print_training = True, device='cuda'):
    """
    
    """
    dict_result = {}
    tensorboard_dict = {}
    
    models = []
    training_losses = []
    val_losses = []
    
    start = time.time()
    
    for epoch in range(nb_epochs):
        model_untrained.train()

        for i, (images, labels) in enumerate(train_loader):  
            images = images.to(device)
            labels = labels.to(device)
            # Clear gradients
            optimizer.zero_grad()
            # Forward propagation
            outputs = model_untrained(images)
            # Calculate cross entropy loss
            training_loss = loss_function(outputs, labels)
            # Backward propagation
            training_loss.backward()
            # Update parameters
            optimizer.step()
            
        if validation_loader != None:
            model_untrained.eval()
            with torch.no_grad():
                for val_images, val_labels in validation_loader:
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)
                    # Forward propagation
                    val_outputs = model_untrained(val_images)
                    # Calculate cross entropy loss
                    val_loss = loss_function(val_outputs, val_labels)
                            

        if print_training:
            if validation_loader == None:
                print (f'Epoch [{epoch + 1} / {nb_epochs}]  --->  Training Loss: {training_loss.item():.4f}')
            else:
                print (f'Epoch [{epoch + 1} / {nb_epochs}]  --->  Training Loss: {training_loss.item():.4f}      Validation loss: {val_loss.item():.4f}')


        mdl = copy.deepcopy(model_untrained) # can as well use .state_dict() but the model class might change
        models.append(mdl)
        training_losses.append(training_loss.item())
        if validation_loader != None:
            val_losses.append(val_loss.item())
        
    # Add stuff to TensorBoard dictionary
    tensorboard_dict['training loss'] = training_losses
    
    end = time.time()
    training_time = end - start
    print('\n\nFinished Training with Total time :  ' + str(end - start) + ' seconds')
    
    dict_result['models'] = models
    dict_result['training_losses'] = training_losses
    dict_result['validation_losses'] = val_losses
    dict_result['tensorboard_dict'] = tensorboard_dict
    dict_result['training time'] = training_time
    
    return dict_result



def select_best_model_loss(dict_):
    """
    """
    models = dict_['models']
    training_losses = dict_['training_losses']
    val_losses = dict_['validation_losses']

    training_losses = np.array(training_losses)
    val_losses = np.array(val_losses)
    
    if len(val_losses) != 0:
        min_loss_idx = np.argmin(val_losses)
        return models[min_loss_idx], val_losses[min_loss_idx]
        
    else:    
        min_loss_idx = np.argmin(training_losses)
        return models[min_loss_idx], training_losses[min_loss_idx]
        



#
#           Main of the Module
# 


if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = torchvision.models.alexnet(pretrained=True)
    
    model.classifier[6] = nn.Linear(4096, 3, bias=True)
    
    model = model.to('cuda')
    
    
    # Hyperparameters
    batch_size = 20
    learning_rate = 0.00001

    
    # Loading Data
    X_train = np.load('Data_Pre_Processed\X_train_128_augmented_preprocessed.npy')
    X_test = np.load('Data_Pre_Processed\X_test_128_not_augmented_preprocessed.npy')
    
    y_train = np.load('Data_Pre_Processed\y_train_128_augmented_preprocessed.npy')
    y_test = np.load('Data_Pre_Processed\y_test_128_not_augmented_preprocessed.npy')
    
    # Reshaping data into (1, 128, 128, 3)  or  for the 2D CNN
    X_train = X_train.reshape(X_train.shape[0], 3, 128, 128)
    X_test = X_test.reshape(X_test.shape[0], 3, 128, 128)

    # Tensors
    X_train  = torch.from_numpy(X_train).float()
    X_test  = torch.from_numpy(X_test).float()
    y_train  = torch.from_numpy(y_train)
    y_test  = torch.from_numpy(y_test)

    # Data loaders
    train = torch.utils.data.TensorDataset(X_train, y_train.long())
    test = torch.utils.data.TensorDataset(X_test, y_test.long())

    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)


    # Training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    # all_images_train_loader  = torch.load('Data_Loaders/all_images_train_loader.pt')
    dict_result = train_model(model, train_loader, optimizer, loss_function, nb_epochs=10, 
                              validation_loader=test_loader, print_training = True, device='cuda')
    
    
    
    exit()
    
    
    # Save best model
    best_model, _ = select_best_model_loss(dict_result)
    torch.save(best_model, 'Models/AlexNet_[2020-11-03__6-00].pt')
    print(_)
    print('\n')
    print(best_model.state_dict())