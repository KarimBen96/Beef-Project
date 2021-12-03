import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.optim import Adam, SGD
from torchsummary import summary
import copy


# Create CNN Model
class CNN_2D_Model_Small(nn.Module):
    def __init__(self):
        super(CNN_2D_Model_Small, self).__init__()
        
        # ReLU Activation function
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
        # Data Normalization
        self.batchnorm2d = nn.BatchNorm2d(2)
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3))
        
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3))
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Fully connected 1
        self.fc1 = nn.Linear(16 * 12 * 12, 16) 
        self.dropout1 = nn.Dropout(0.3)
        
        # Fully connected 2
        self.fc2 = nn.Linear(16, 16) 
        self.dropout2 = nn.Dropout(0.3)
        
        # Fully connected 3
        self.fc3 = nn.Linear(16, 3) 
        
    def forward(self, x):
        # out = self.batchnorm3d(x)
        
        # Set 1
        out = self.cnn1(x)
        out = self.relu(out)
        
        # Set 2
        out = self.cnn2(out)
        out = self.relu(out)
        #out = self.dropout1(out)
        
        # Flatten
        out = self.flatten(out)
        flatten_image = copy.copy(out)
        
        # Dense 1
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        # Dense 2
        """
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        """
        # Dense 3
        out = self.fc3(out)
        out = self.softmax(out)
        
        return out, flatten_image
    
    
    


#   
#               Main of the Module
#



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_for_summary = CNN_2D_Model_Small().to(device)
    # model_summary = summary(model_for_summary, (1, 128, 128, 3))
    model_summary = summary(model_for_summary, (3, 16, 16))
    

