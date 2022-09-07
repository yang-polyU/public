import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2

import shapes
import train
import test

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

    opt = parser.parse_args()
    print(opt)


if __name__ == '__main__':
    main()

