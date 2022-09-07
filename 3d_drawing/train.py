import torch
import torch.nn as nn                                            #achieve all kinds of layers
import torch.nn.functional as F                                  #achieve all kinds of functions
import torch.optim as optim                                      #achieve all packages of optim algorithms
from torchvision import datasets, transforms

#the process of training mainly includes loading training set, forward propagation,
#calculating loss, backward propagation, updating parameters
def train(model, device, train_loader, optimizer, epoch):
    model.train()                                                #set the model to training mode
    for batch_idx, (data, label) in enumerate(train_loader):     #iterate over a batch of data from the data loader
        data, label = data.to(device), label.to(device)          #store the data to cpu or gpu
        optimizer.zero_grad()                                    #clear all the optimaized gradient
        output = model(data)                                     #feed the data and get the output through forward propagation
        loss = nn.CrossEntropyLoss(output, label)                #calculate the loss by using loss func
        loss.backward()                                          #backforward propagation
        optimizer.setp()                                         #update parameters
        if batch_idx % 20 == 0:                                  #output training log
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    torch.save(model, "trainingModel.pth")                       #save net model
