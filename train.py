# Imports here
#Not able to upload the package 

import torch
from torch import optim, nn 
from torchvision import datasets,models,transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import argparse
#checkpoint mein arch bhi add karna hai
parser = argparse.ArgumentParser(
    description='train a neural network',
)

parser.add_argument("data_dir",help="data directory")
parser.add_argument("--save_dir",help="save  directory", default="./himanshu.pth")
parser.add_argument("--learning_rate",default = 0.001 ,type=int,help="learning rate")
parser.add_argument("--epochs",default = 20 ,type=int,help="epochs")
parser.add_argument("--gpu",default = 'cpu' ,help="GPU/CPU")
parser.add_argument("--arch",default = 'densenet121' ,help="pretrained network")
parser.add_argument("--hidden_units",default = 512 ,type=int,help="number of hidden units ")  
args = parser.parse_args()
print(args)
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean =[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
data_transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean =[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_datasets =  datasets.ImageFolder(train_dir,transform = train_transforms)
test_datasets =  datasets.ImageFolder(test_dir,transform = data_transform)
valid_datasets =  datasets.ImageFolder(valid_dir,transform = data_transform)


# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloaders = torch.utils.data.DataLoader(train_datasets,batch_size=32,shuffle=True)
testloaders = torch.utils.data.DataLoader(test_datasets,batch_size=32,shuffle=True)
validloaders = torch.utils.data.DataLoader(valid_datasets,batch_size=32,shuffle=True)
modelname = args.arch
if modelname ==  'vgg16':
    model = models.vgg16(pretrained = True)
model = models.densenet121(pretrained = True)
for params in model.parameters():
    params.requires_grad = False
    
#hidden units

Classifier = nn.Sequential(OrderedDict([
                     ('fc1',nn.Linear(1024,args.hidden_units,bias=True)),
                     ('relu', nn.ReLU()),
                     ('dropout', nn.Dropout(p=0.2)),
                     ('fc2',nn.Linear(args.hidden_units,256)),
                     ('relu',nn.ReLU()),
                     ('dropout',nn.Dropout(p=0.2)),
                     ('fc3',nn.Linear(256,102,bias=True)),
                     ('output',nn.LogSoftmax(dim=1))
                     ]))
model.classifier = Classifier
for param in model.classifier.parameters():
    param.requires_grad = True
    
device = args.gpu
print("Current device" , device)
#device =  torch.device("cuda" if  torch.cuda.is_available() else "cpu")
model.to(device)
epochs = args.epochs
steps = 0  
print_every = 10 
optimizer = optim.Adam(model.classifier.parameters(),lr=args.learning_rate)
criterion =  nn.NLLLoss()
accuracy1 = 0    


train_losses, test_losses = [], []
print("Trainig process ")
for e in range(epochs):
    running_loss = 0
    accuracy1 = 0
    for images, labels in trainloaders:
        images,labels = images.to(device) , labels.to(device)
        optimizer.zero_grad()
        #log_ps is  used as output so neural network try to match labels with log_ps using backpropogation
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy1 += torch.mean(equals.type(torch.FloatTensor))
        running_loss += loss.item()
        
    else:
        test_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for images, labels in testloaders:
                images,labels = images.to(device) , labels.to(device)
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)
                #here we are using ps to calculate the accuracy.
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                #why we do not use log_ps to calculate the accuracy
        train_losses.append(running_loss/len(trainloaders))
        test_losses.append(test_loss/len(testloaders))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloaders)),
              "Test Loss: {:.3f}.. ".format(test_loss/len(testloaders)),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloaders)),                                                       
              "Train Accuracy: {:.3f}".format(accuracy1/len(trainloaders))
                                                                         )
        
class_to_idx = train_datasets.class_to_idx
model.class_to_idx = class_to_idx
model.name =  modelname
checkpoint = {
            'architecture': model.name,
            'classifier': model.classifier,
            'class_to_idx': model.class_to_idx,
            'state_dict': model.state_dict
}
savedir = args.save_dir
print("Saved Checkpoint")
torch.save(checkpoint,savedir)
