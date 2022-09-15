import torch
import torch.nn as nn                                            #achieve all kinds of layers
import torch.nn.functional as F                                  #achieve all kinds of functions
import torch.optim as optim                                      #achieve all packages of optim algorithms
from torchvision import datasets, transforms

def test(model, device, test_loader):
    model.eval()                                                 #set the model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():                                        #forbid gradient calculation
        for data, label in test_loader:                          #iterate over a batch of data from the data loader
            data, label = data.to(device), label.to(device)      #store the data to cpu or gpu
            output = model(data)
            #sum up batch loss
            test_loss += F.nll_loss(output, label, reduction = 'sum').item()
            pred = output.max(1, keepdim = True)[1]              #get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item() #Counting the number of correct predictions

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

