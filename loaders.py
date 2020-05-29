import os 
import config
from imutils import paths
import torch.utils.data as data 
from torchvision import datasets, transforms


# Transform the data to torch tensors and normalize it 

transform_idc = transforms.Compose([transforms.RandomResizedCrop(48),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.8023, 0.6129, 0.7238], 
                                                           std=[0.0158, 0.0219, 0.0178])]) 


# Prepare training set, validation set and testing set
trainset = datasets.ImageFolder(config.TRAIN_PATH, transform=transform_idc)
valset = datasets.ImageFolder(config.VAL_PATH, transform=transform_idc)
testset = datasets.ImageFolder(config.TEST_PATH, transform=transform_idc )

#parameters 
params = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 4}

# Prepare training loader and testing loader (makes datasets iterable)

trainloader = data.DataLoader(trainset, **params)
valloader = data.DataLoader(valset, **params)
testloader = data.DataLoader(testset, **params)
