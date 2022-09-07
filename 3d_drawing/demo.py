import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn                                 #achieve all kinds of layers
import torch.nn.functional as F                       #achieve all kinds of functions
import torch.optim as optim                           #achieve all packages of optim algorithms
from torchvision import datasets, transforms

from shapes import Sphere, Cube, Pyramid, Frustum
import train
import test
from network import Net

# 3D reconstruction
# graph = Sphere(1, 2, 3, 10)
# graph.reconstruction()

# graph = Cube(10, 10, 10, 10, 10, 10)
# graph.reconstruction()

# graph = Pyramid(0, 0, 0, 10, 10, 10, 5, 5)
# graph.reconstruction()

# graph = Frustum(0, 0, 0, 20, 20, 20, 10, 10, 5, 5)
# graph.reconstruction()

def main():
# 2D img processing
    # img = cv2.imread('D:\\work_shop\\3d_drawing\\part_img.bmp', 1)#rgb scale


    # print("img shape : " + str(img.shape))
    # print(img)
    # cv2.imshow('sample', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    parser = argparse.ArgumentParser(description='CNN Example')
    parser.add_argument("--stage", type = str, default = 'train', help = "is train or test")
    parser.add_argument("--epochs", type = int, default = 30, help = "the number of epochs of training")
    parser.add_argument("--batch_size", type = int, default = 128, help = "size of the batches")
    parser.add_argument("--lr", type = float, default = 0.001, help = "SGD: learning rate")
    parser.add_argument("--momentum", type = float, default = 0.9, help = "SGD: momentum")
    parser.add_argument("--img_size", type = tuple, default = (28,28), help = "size of each image dimension")
    parser.add_argument("--channels", type = int, default = 1, help = "number of image channels")
    parser.add_argument("--predictImg", type = str, default = '', help = "image need to be predicted")

    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")              #set using cpu or gpu

    if opt.stage == "train":                                                           #in train stage
        dataLoader = torch.utils.data.DataLoader()
        model = Net(opt.channels)
        optimizer = optim.SGD(model.parameters(), lr = opt.lr, momentum = opt.momentum)
        train(model, device, dataLoader, optimizer, opt.epochs)
    elif opt.stage == "test":                                                          #in test stage
        testLoader = torch.utils.data.DataLoader()
        model = torch.load('trainingModel.pth')
        test(model, device, testLoader)



if __name__ == '__main__':
    main()

