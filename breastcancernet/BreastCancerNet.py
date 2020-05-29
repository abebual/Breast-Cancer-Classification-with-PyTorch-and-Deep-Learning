# import the necessary packages 

#container
from torch.nn import Sequential
from torch.nn import Module

#convolution layers
from torch.nn import Conv2d

#pooling layers
from torch.nn import MaxPool2d

#Activations
from torch.nn import ReLU
from torch.nn import Softmax

#normalization layers
from torch.nn import BatchNorm2d

#Linear Layers
from torch.nn import Linear

#Dropout layers
from torch.nn import Dropout

#loss function 
from torch.nn import CrossEntropyLoss 
#This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

#Utility functions
from torch.nn import Flatten


#Build CNN BreastCancerNet in PyTorch
#Adopted from CancerNet model in Keras by DataFlair Team

class BreastCancerNet(Module):
    def __init__(self):
        super(BreastCancerNet, self).__init__()
        
       # Declare all the layers for feature extraction
        self.features = Sequential(Conv2d(in_channels=3,
                                          out_channels=32,
                                          kernel_size = (3, 3), 
                                          padding=1),
                                   ReLU(inplace=True),
                                   BatchNorm2d(32),
                                   MaxPool2d(3,3),
                                   Dropout(0.23),
                                   Conv2d(in_channels=32, 
                                          out_channels=64, 
                                          kernel_size = (3, 3), 
                                          padding=1),
                                   ReLU(inplace=True),
                                   BatchNorm2d(64),
                                   MaxPool2d(3,3),
                                   Dropout(0.25),
                                   Conv2d(in_channels=64, 
                                          out_channels=128, 
                                          kernel_size = (3, 3), 
                                          padding=1),
                                   ReLU(inplace=True),
                                   BatchNorm2d(128),
                                   MaxPool2d(3,3),
                                   Dropout(0.25))
        
                                   
        # Declare all the layers for classification (fully connected layers)
        self.classifier = Sequential(
            Linear(4096, 6912),
            ReLU(inplace=True),
            Dropout(0.5),
            Linear(6912, 6912),
            ReLU(inplace=True),
            Linear(6912, 2))
        
        #self._initialize_weights()
                 
    def  forward(self, x): 
                 #Apply the feature extract in the input
                 x = self.features(x)
                 #squeeze the three spatial dimenstions in one 
                 x = x.view(-1, 4096)
                 x = self.classifier(x)
                 return x