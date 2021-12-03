import numpy as np
import pickle
import collections
import scipy.io
import time
import datetime
import copy

import albumentations as A
import cv2

from data_pre_processing import *





def augment_image(transform, image):
    """
    """
    
    new_image = copy.deepcopy(image)
    new_image = transform(image=new_image)['image']

    return new_image



def augment_X(X_train, transforms):
    """
    - Only for 4 times the length of the original data
    - X_train and X_train_augmented are 4 dim arrays
    """
    
    X_train_augmented = np.zeros((X_train.shape[0] * 4, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    
    ii = 0
    for i in range(X_train.shape[0]):
        image_to_augment = X_train[i, :, :, :]
        X_train_augmented[i + ii, :, :, :] = image_to_augment

        for j in range(3):
            transform = None
            transformed_image = None
            transform = transforms[j]
            transformed_image = augment_image(transform, image_to_augment)
            X_train_augmented[i + ii + j + 1, :, :, :] = transformed_image
        ii += 3
        
    return X_train_augmented



def augment_Y(y_train):
    """
    - Only for 4 times the length of the original data
    """
    
    y_train_augmented = np.zeros((y_train.shape[0] * 4), dtype=np.uint8)
    
    ii = 0
    for i in range(y_train.shape[0]):
        y_train_augmented[i + ii] = y_train[i]
        y_train_augmented[i + ii + 1] = y_train[i]
        y_train_augmented[i + ii + 2] = y_train[i]
        y_train_augmented[i + ii + 3] = y_train[i]
        ii += 3

    return y_train_augmented







#
#           Main of the Module
#




if __name__ == '__main__':
    
    # Loading Data
    X, y = load_adundance_maps('Data_Raw/Abundance_Maps/Ab_Maps_Beef_128.mat', 'Data_Raw/all_labels.npy')
    X = reshape_abandance_maps_4d(X, (186, 3, 128, 128))
    
    
    # Save some data for Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    np.save('Data_Raw/Data_Augmented/X_test_128_not_augmented', X_test)
    np.save('Data_Raw/Data_Augmented/y_test_128_not_augmented', y_test)
    
    
    
    
    # Define transforms
    transform_1 = A.HorizontalFlip(always_apply=True)
    transform_2 = A.VerticalFlip(always_apply=True)
    transform_3 = A.Compose([
        A.VerticalFlip(always_apply=True),
        A.HorizontalFlip(always_apply=True)
    ])
    all_transforms = [transform_1, transform_2, transform_3]
    
    
    # Augmenting Train Data
    X_train_augmented = augment_X(X_train, all_transforms)
    y_train_augmented = augment_Y(y_train)
    
    print(X_train.shape)
    print(X_train_augmented.shape)
    print(collections.Counter(y_train))
    print(collections.Counter(y_train_augmented))
    
    
    # Save Augmented Train Data
    np.save('Data_Raw/Data_Augmented/X_train_128_augmented', X_train_augmented)
    np.save('Data_Raw/Data_Augmented/y_train_128_augmented', y_train_augmented)

    
    
    