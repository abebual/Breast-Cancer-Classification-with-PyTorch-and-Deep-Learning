# Breast Cancer Classification with PyTorch
In this project, we’ll build a classifier to train on 80% of a breast cancer histology image dataset. Of this, we’ll keep 10% of the data for validation. Using PyTorch, we’ll define a CNN (Convolutional Neural Network), call it BreastCancerNet, and train it for 40 epochs on our images and compute accuracy on the validation set. We’ll then derive a confusion matrix to analyze the performance of the model to detect IDC breast cancer in histopathological images. Two classes will be detected, 0 and 1: 0 denotes absence of IDC and 1 denotes presence of IDC.


## Requirements
`requirements.txt`
+ torch
+ torchvision

## Dataset and Data Setup
Download Breast Cancer Histology Image Dataset from [kaggle](https://www.kaggle.com/paultimothymooney/breast-histopathology-images/source). 

To set up idc datasets in PyTorch open `config.py` and change path to datasets. `dataset.py` to copy the downloaded dataset to the datasets/origianl folder, we declare the path for the new directory (datasets/idc), and the paths for the training, validation, and testing directories using the base path. We also declare that 80% of the entire dataset will be used for training, and of that, 10% will be used for validation.Next, we’ll import from config, imutils, random, shutil, and os. Then we will grab all the *originalPaths* for our dataset and randomly suffle them.

Then, we calculate an *index* by multiplying the length of this list by 0.8 so we can slice this list to get sublists for the training and testing datasets. Next, we further calculate an index saving 10% of the list for the training dataset for validation (*valPaths*) and keeping the rest for training itself (*trainPaths*). Then, we defines a list with tuples called *datasets*. 

`loaders.py` - Transform the data to torch tensors and normalize it and prepare training loader and testing loader (makes datasets iterable).

## Build BreastCancerNet CNN Model
`BreastCancerNet.py`

## Train, validate, and test Model
`main.py`

## Citations
[1] American Cancer Society. Cancer Facts & Figures 2020. Available at: https://www.cancer.org/content/dam/cancer-org/research/cancer-facts-and-statistics/annual-cancer-facts-and-figures/2020/cancer-facts-and-figures-2020.pdf.  
[2] Veta M, van Diest PJ, Willems SM, Wang H, Madabhushi A, Cruz-Roa A, et al. Assessment of algorithms for mitosis detection in breast cancer histopathology images. Med Image Anal. 2015;20:237– 48. [PubMed: 25547073]  
[3] Roux L, Racoceanu D, Loménie N, Kulikova M, Irshad H, Klossa J, et al. Mitosis detection in breast cancer histological images An ICPR 2012 contest. J Pathol Inform. 2013;4:8. [PMCID: PMC3709417] [PubMed: 23858383]  
[4] Ciresan DC, Giusti A, Gambardella LM, Schmidhuber J. Mitosis detection in breast cancer histology images with deep neural networks. Med Image Comput Comput Assist Interv. 2013;16(Pt 2):411–8. [PubMed: 24579167]  
[5] Cruz-Roa A, Basavanhally A, González F, Gilmore H, Feldman M, Ganesan S, et al. Automatic detection of invasive ductal carcinoma in whole slide images with convolutional neural networks. SPIE Medical Imaging. 2014;9041:904103-904103-15.  



