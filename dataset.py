import os 
from breastcancernet import config
from imutils import paths
import random, shutil

#declare the path to the input dataset (datasets/original)
os.chdir('C:\\Users\\abebu\\Dropbox\\Data Science\\DeepLearning\\Capstone Project 2\\breast-cancer-classification')
mkdir datasets\original

#grab all the originalPaths for our dataset and randomly suffle them
originalPaths=list(paths.list_images(config.INPUT_DATASET))
random.seed(7)
random.shuffle(originalPaths)

#calculate an index by multiplying the length of this list by 0.8 so we can slice this list to get sublists for the training and testing datasets
index=int(len(originalPaths)*config.TRAIN_SPLIT)
trainPaths=originalPaths[:index]
testPaths=originalPaths[index:]

#calculate an index saving 10% of the list for the training dataset for validation and keep the rest for training 
index=int(len(trainPaths)*config.VAL_SPLIT)
valPaths=trainPaths[:index]
trainPaths=trainPaths[index:]

#defines a list with tuples called datasets and organize originalPaths into training, validation, and testing sets
datasets= [('training', trainPaths, config.TRAIN_PATH),
          ('validation', valPaths, config.VAL_PATH),
          ('testing', testPaths, config.TEST_PATH)
]

#build the path to the resulting image and copy image into its destination

for (setType, originalPaths, basePath) in datasets:
    print(f'Building {setType} set')
    
    if not os.path.exists(basePath):
        print(f'Building directory {basePath}')
        os.makedirs(basePath)
        
    for path in originalPaths:
        file=path.split(os.path.sep)[-1]
        label=file[-5:-4]
        
        labelPath=os.path.sep.join([basePath, label])
        if not os.path.exists(labelPath):
            print(f'Building directory {labelPath}')
            os.makedirs(labelPath)
            
        newPath=os.path.sep.join([labelPath, file])
        shutil.copy2(path, newPath)