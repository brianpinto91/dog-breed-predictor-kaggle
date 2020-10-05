'''******************************************************************************
*Programming language: Python
*Filename: trainer.py
*Description: This module is used to train a dog classifier deep learning model using Pytorch.
              The dataset was provided in a Kaggle competition
*Author: Brian Pinto
*Version: 1.0
*Date: 07.11.2020
##############################################################################'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import utils

Train = False
Submit = False
LOAD_TRIANED_MODEL = True

data_dir = "../input"
train_fol = os.path.join(data_dir, "train")
test_fol = os.path.join(data_dir, "test")
train_labels = pd.read_csv(os.path.join(data_dir,"labels.csv"))
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

BREEDS = list(train_labels['breed'].unique())
BREEDS.sort()
BREEDS_TO_CLASS = {x:y for x,y in zip(BREEDS,range(len(BREEDS)))}
CLASS_TO_BREED = {y:x for y,x in enumerate(BREEDS_TO_CLASS)}

#create a dataframe for trainset similar to submission dataframe
train_labels['class'] = 1
labels_matrix = train_labels.pivot('id','breed','class').reset_index().fillna(0)

if Train == True:
    #split the training data into train and validation
    train_labels, val_labels = np.split(labels_matrix, [int(len(labels_matrix)*0.8)], axis=0)

    #Pytorch dataset and dataloader

    train_ds = utils.DogDataset(train_fol, train_labels)
    val_ds = utils.DogDataset(train_fol, val_labels)

    train_loader = DataLoader(train_ds, batch_size=30, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=30, shuffle=False)

    dataloaders = {'train':train_loader,'val':val_loader}

    model = utils.DogBreedPredictor(pretrained = True)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001) #initialize the optimizer
    loss_criteria = torch.nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1, last_epoch=-1)

if Submit == True:
    model, traing_curve, val_curve = utils.train_model(model, dataloaders, optimizer, loss_criteria, scheduler, 25, True)
    sub_ds = utils.DogDataset(test_fol, sample_sub, training=False)
    sub_dl = DataLoader(sub_ds, batch_size=30, shuffle=False)
    sub_df = utils.submit_model(model, sub_dl)
    sub_df_names = sub_df.idxmax(axis=1)
    sub_df_names = pd.DataFrame({'id': sub_df_names.index, 'breed': sub_df_names.values})
    sub_df.to_csv("../output/submission.csv", index=True)
    sub_df_names.to_csv("../output/test_predictions.csv")

if LOAD_TRIANED_MODEL==True:
    vis_df = labels_matrix.sample(4)
    vis_ds = utils.DogDataset(train_fol, vis_df)
    vis_dl = DataLoader(vis_ds, batch_size=len(vis_ds), shuffle=False)

    model = utils.DogBreedPredictor(False)
    model.load_state_dict(torch.load("../input/saved-models/model1.pt", map_location = torch.device('cpu')))
    model.eval()


    for image_names, imgs, lbls in vis_dl:
        vis_imgs = list(image_names)
        with torch.no_grad():
            outputs = model(imgs)
            _, preds = torch.max(outputs,1)
            preds = preds.data.tolist()
            lbls = lbls.data.tolist()


   
    fig, axs = plt.subplots(1,4)
    for i in range(4):
        axs[i].imshow(Image.open(os.path.join(train_fol,vis_imgs[i])))
        axs[i].set_title("A: " + CLASS_TO_BREED[lbls[i]] + "\n" + "P: " + CLASS_TO_BREED[preds[i]])
          
