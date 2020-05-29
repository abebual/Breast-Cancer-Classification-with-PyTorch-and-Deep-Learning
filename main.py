# import the necessary packages
import torch
import torch.nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.optim import Adagrad
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy
from torch.nn.utils import convert_parameters
from onecyclelr import OneCycleLR
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from breastcancernet import BreastCancerNet
import config
import loaders
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# initialize our number of epochs, initial learning rate, and batch size
num_epochs=40; lr=1e-2; batch_size=32; num_classes=2

#initialize dataloaders
trainloader = loaders.trainloader
valloader = loaders.valloader
testloader = loaders.testloader

#We would like to use GPU for training if possible to speed up training process
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# initialize our CancerNet model
model= BreastCancerNet.BreastCancerNet()

#model parameters 
for param in model.parameters():
    print(type(param), param.size())

#function to train a batch of IDC images
def train(trainloader, model, criterion, optimizer, scheduler):
    total_loss = 0.0
    size = len(trainloader.dataset)
    num_batches = size // trainloader.batch_size
    model.train()
    for i, (images, labels) in enumerate(trainloader):
        print(f"Training: {i}/{num_batches}", end="\r")
        
        scheduler.step()
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images) # forward pass
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        loss.backward()  # backprogagation
        optimizer.step()
        
    return total_loss / size

#function to compute the accuracy on the validation set.
def validate(valloader, model, criterion):
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_loss = 0.0
        size = len(valloader.dataset)
        num_batches = size // valloader.batch_size
        for i, (images, labels) in enumerate(valloader):
            print(f"Validation: {i}/{num_batches}", end="\r")
            
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            total_correct += torch.sum(preds == labels.data)
            total_loss += loss.item() * images.size(0)
            
        return total_loss / size, total_correct.double() / size

#main training loop 
target_size=torch.rand((48,48), requires_grad=False)
input_size=torch.rand((48,48), requires_grad=False)

def fit(model, num_epochs, trainloader, valloader):
    criterion = binary_cross_entropy(input_size, target_size) 
    optimizer = Adagrad(model.parameters(), lr=lr,lr_decay=lr/num_epochs)
    scheduler = OneCycleLR(optimizer, lr_range=(lr,1.), num_steps=1000)
    print("epoch\ttrain loss\tvalid loss\taccuracy")
    for epoch in range(num_epochs):
        train_loss = train(trainloader, model, criterion, optimizer, scheduler)
        valid_loss, valid_acc = validate(valloader, model, criterion)
        print(f"{epoch}\t{train_loss:.5f}\t\t{valid_loss:.5f}\t\t{valid_acc:.3f}")


#train for 40 epochs and print the training loss, validation loss, and accuracy improve with each epoch
model = model.to(device)
fit(model, 40, trainloader, valloader)

#show ROC
def results(model, valloader):
    model.eval()
    preds = []
    actual = []
    with torch.no_grad():
         for images, labels in valloader:
            outputs = model(images.to(device))
            preds.append(outputs.cpu()[:,1].numpy())
            actual.append(labels.numpy())
    return np.concatenate(preds), np.concatenate(actual)

preds, actual = results(model, valloader)
fpr, tpr, _ = roc_curve(actual, preds)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, label=f"ROC curve (area = {auc(fpr, tpr):.3f})")
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristics'); plt.legend()

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# show formatted classification report
print(classification_report(testloader.classes, pred_indices, target_names=testloader.class_indices.keys()))

# compute the confusion matrix and and use it to derive the raw accuracy, sensitivity, and specificity
cm=confusion_matrix(testloader.classes,pred_indices)
total=sum(sum(cm))
accuracy=(cm[0,0]+cm[1,1])/total
specificity=cm[1,1]/(cm[1,0]+cm[1,1])
sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])

# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print(f'Accuracy: {accuracy}')
print(f'Specificity: {specificity}')
print(f'Sensitivity: {sensitivity}')

# plot the training loss and accuracy
N = num_epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), M.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), M.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), M.history["acc"], label="train_acc")
plt.plot(np.arange(0,N), M.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on the IDC Dataset")
plt.xlabel("Epoch No.")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('plot.png')