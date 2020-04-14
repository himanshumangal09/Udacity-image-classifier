# Imports here
import torch
from torch import optim, nn 
from torchvision import datasets,models,transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from collections import OrderedDict
import json
import argparse
from PIL import Image

parser = argparse.ArgumentParser(
    description='predict  a neural network',
)

parser.add_argument("i_path",help="path for image")
parser.add_argument("filecheck",help="load the checkpoint",)
parser.add_argument("--top_k",default = 5 ,type=int,help="top classes")
parser.add_argument("--category_names",default = 'cat_to_name.json' ,help="Json file for mapping")
parser.add_argument("--gpu",default = 'cpu' ,help="GPU/CPU")



def load_checkpoint(checkpoint):
    """
    Loads deep learning model checkpoint.
    """
    
    # Load the saved file
    checkpoint = torch.load(checkpoint)
    
    # Download pretrained model
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True);
    model = models.densenet121(pretrained=True);
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): 
        param.requires_grad = False
    
    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict = checkpoint['state_dict']
    return model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    img_r = img.resize((255,255))
    l = (255 - 224)/2
    t = (255-224)/2
    r = (255 + 224)/2
    b = (255+224)/2
    img_c  = img_r.crop((l,t,r,b))
    #print(img_c.shape)
    np_image = np.array(img_c)
    np_image  = np_image/ 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    new_np_image = (np_image - mean)/std
    fin_img = new_np_image.transpose((2,0,1))
    return fin_img

    # TODO: Process a PIL image for use in a PyTorch model

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    pytorch_np_image = process_image(image_path)    
    # Changing from numpy to pytorch tensor
    pytorch_tensor = torch.tensor(pytorch_np_image)
    pytorch_tensor = pytorch_tensor.float()
    
    # Removing RunTimeError for missing batch size - add batch size of 1 
    pytorch_tensor = pytorch_tensor.unsqueeze(0)
    
    # Run model to make predictions
    model.eval()
    model.to('cpu')
    LogSoftmax_predictions = model(pytorch_tensor)
    predictions = torch.exp(LogSoftmax_predictions)
    
    # Identify top predictions and top labels
    top_preds, top_indi = predictions.topk(5)
    top_indi = top_indi.tolist () [0]
    print("Propabilities " , top_preds)
    mapping = {val: key for key, val in
                model.class_to_idx.items()
                }

    classes = [mapping [item] for item in top_indi]
    classes = np.array (classes) 
    print("classes",classes)
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)  
    class_names = [cat_to_name [item] for item in classes]
    print(class_names)
args = parser.parse_args()
image_path = args.i_path
checkpoint = args.filecheck
model = load_checkpoint(checkpoint) 
model.to("cpu")
predict(image_path,model)
