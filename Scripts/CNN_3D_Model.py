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
class CNN_3D_Model(nn.Module):
    def __init__(self):
        super(CNN_3D_Model, self).__init__()
        
        # ReLU Activation function
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
        # Data Normalization
        self.batchnorm3d = nn.BatchNorm3d(3)
        
        # Convolution 1
        self.cnn1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 1))
     
        # Convolution 2
        self.cnn2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))
        
        # Convolution 3
        self.cnn3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        
        # Max pool 3
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))
        
        # Convolution 4
        self.cnn4 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        
        # Max pool 4
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 1))
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Fully connected 1
        #self.fc1 = nn.Linear(16 * 30 * 30, 32) 
        self.fc1 = nn.Linear(32 * 14 * 14 * 3, 32) 
        self.dropout1 = nn.Dropout(0.3)
        
        # Fully connected 2
        self.fc2 = nn.Linear(32, 16) 
        self.dropout2 = nn.Dropout(0.3)
        
        # Fully connected 3
        self.fc3 = nn.Linear(32, 3) 
        
    def forward(self, x):
        # out = self.batchnorm3d(x)
        
        # Set 1
        out = self.cnn1(x)
        out = self.relu(out)
        out = self.maxpool1(out)
        
        # Set 2
        out = self.cnn2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        out = self.dropout1(out)
        
        # Set 3
        out = self.cnn3(out)
        out = self.relu(out)
        out = self.maxpool3(out)
        out = self.dropout1(out)
        
        # Flatten
        #out = out.view(out.size(0), -1)
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

    model_for_summary = CNN_3D_Model().to(device)
    model_summary = summary(model_for_summary, (1, 128, 128, 3))
    #model_summary = summary(model_for_summary, (3, 128, 128, 1))
    

