'''******************************************************************************
*Programming language: Python
*Filename: trainer.py
*Description: This module has the data preprocessing and training utilities
*Author: Brian Pinto
*Version: 1.0
*Date: 07.05.2020
##############################################################################'''

import os
import pandas as pd
import numpy as np
import datetime as dt
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms, models

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.476, 0.452, 0.392],
                             std=[0.235, 0.231, 0.229])
    ])

class DogDataset(Dataset):
    def __init__(self, image_folder, labels_matrix, transformers = data_transform, training = True):
        self.image_folder = image_folder
        self.labels_matrix = labels_matrix
        self.tf = transformers
        self.training = training
        
    def __len__(self):
        return self.labels_matrix.shape[0]

    def __getitem__(self, idx):
        image_name_i = self.labels_matrix.iloc[idx]['id'] + '.jpg'
        image_i  = Image.open(os.path.join(self.image_folder, image_name_i))
        image_i = self.tf(image_i)
        if self.training == True:
            label_i = self.labels_matrix.iloc[idx,1:].values.argmax()
            return image_name_i, image_i, label_i
        else:
            return image_name_i, image_i

def DogBreedPredictor(pretrained = True):
    model = models.resnet50(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    in_fea = model.fc.in_features
    model.fc = torch.nn.Linear(in_fea, 120)    
    return model

def train_model(model, data_loaders, optimizer, loss_criteria, scheduler, epochs, use_gpu=True):
    begin = dt.datetime.now() #Start the timer
    training_loss = [] #to plot loss curve
    val_loss = [] #to plot loss curve
    if use_gpu and torch.cuda.is_available():
        print("Using GPU")
        model = model.cuda()
    best_model_wts = model.state_dict()
    best_val_acc = 0.0
    for epoch in range(epochs):
        #train
        train_cum_loss_epoch = 0.0
        train_correct_predictions = 0.0
        model.train()
        for _, inputs, labels in data_loaders['train']:
            if use_gpu and torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_criteria(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predictions = torch.max(outputs,1)
            train_correct_predictions+=torch.sum(predictions==labels).item()
            train_cum_loss_epoch+=loss.item()
        scheduler.step()
        
        #validation
        val_cum_loss_epoch = 0.0
        val_correct_predictions = 0.0
        model.eval()
        for _, inputs, labels in data_loaders['val']:
            if use_gpu and torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            _, predictions = torch.max(outputs,1)
            val_correct_predictions+=torch.sum(predictions==labels).item()
            val_cum_loss_epoch+=loss_criteria(outputs, labels).item()
            
        #metrics
        train_mean_epoch_loss = train_cum_loss_epoch / data_loaders['train'].batch_size
        val_mean_epoch_loss = val_cum_loss_epoch / data_loaders['val'].batch_size
        train_accuracy = train_correct_predictions / len(data_loaders['train'].dataset)
        val_accuracy = val_correct_predictions / len(data_loaders['val'].dataset)
        training_loss.append(train_mean_epoch_loss)
        val_loss.append(val_mean_epoch_loss)
        #print
        print('Epoch [{}/{}] train loss: {:.4f} train acc: {:.4f} ' 
              'val loss: {:.4f} val acc: {:.4f}'.format(
                epoch + 1 , epochs,
                train_mean_epoch_loss, train_accuracy, 
                val_mean_epoch_loss, val_accuracy))
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            opt_epoch = epoch + 1
            best_model_wts = model.state_dict()
    print("Running time: ", dt.datetime.now()-begin)
    print('Best val Acc: {:4f} and optimal epoch: {}'.format(best_val_acc, opt_epoch))
    #load best weights
    model.load_state_dict(best_model_wts)
    return model, training_loss, val_loss

def submit_model(model, data_loader, indices, columns, use_gpu=True):
    sub_df = pd.DataFrame(index = indices, columns=columns)
    sub_val = []
    if use_gpu and torch.cuda.is_available():
        print("Using GPU")
        model = model.cuda()
    model.eval()    
    with torch.no_grad():
        for _, inputs in data_loader:
            if use_gpu and torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = model(inputs)
            outputs = torch.nn.functional.softmax(outputs,dim=1)
            sub_val.append(outputs.data.cpu().numpy())
    sub_val = np.concatenate(sub_val)
    sub_df.loc[:,:] = sub_val
    return sub_df
