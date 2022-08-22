import numpy as np
import matplotlib.pyplot as plt

class Sphere():
    def __init__(self, x, y, z, radius):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
    
    def reconstruction(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection = "3d")

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        center_x = self.radius * np.outer(np.cos(u), np.sin(v)) + self.x
        center_y = self.radius * np.outer(np.sin(u), np.sin(v)) + self.y
        center_z = self.radius * np.outer(np.ones(np.size(u)), np.cos(v)) + self.z

        ax.plot_surface(center_x, center_y, center_z)
        plt.show()

class Cylinder():
    def __init__(self):
        pass

    def reconstruction(self):
        pass

class Cube():
    def __init__(self, x, y, z, length, width, height):
        self.x = x
        self.y = y
        self.z = z
        self.length = length
        self.width = width
        self.height = height

    def reconstruction(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection = "3d")

        xx = np.linspace(self.x, self.x + self.length, 2)
        yy = np.linspace(self.y, self.y + self.width, 2)
        zz = np.linspace(self.z, self.z + self.height, 2)

        xx2, yy2 = np.meshgrid(xx, yy)
        ax.plot_surface(xx2, yy2, np.full_like(xx2, self.z))
        ax.plot_surface(xx2, yy2, np.full_like(xx2, self.z + self.height))

        yy2, zz2 = np.meshgrid(yy, zz)
        ax.plot_surface(np.full_like(yy2, self.x), yy2, zz2)
        ax.plot_surface(np.full_like(yy2, self.x + self.length), yy2, zz2)

        xx2, zz2 = np.meshgrid(xx, zz)
        ax.plot_surface(xx2, np.full_like(yy2, self.y), zz2)
        ax.plot_surface(xx2, np.full_like(yy2, self.y + self.width), zz2)

        plt.show()

class Pyramid():
    def __init__(self, x, y, z, length, width, height, p_x, p_y):
        self.x = x
        self.y = y
        self.z = z
        self.length = length
        self.width = width
        self.height = height
        self.p_x = p_x
        self.p_y = p_y

    def reconstruction(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection = "3d")

        xx = np.linspace(self.x, self.x + self.length, 2)
        yy = np.linspace(self.y, self.y + self.width, 2)

        #surface 1
        xx1, yy1 = np.meshgrid(xx, yy)
        ax.plot_surface(xx1, yy1, np.full_like(xx1, self.z))

        #surface 2
        x2 = [self.x, self.p_x, self.x]
        y2 = [self.y, self.p_y, self.y + self.width]
        z2 = [self.z, self.z + self.height, self.z]
        ax.plot_trisurf(x2, y2, z2)

        #surface 3
        x3 = [self.x, self.p_x, self.x + self.length]
        y3 = [self.y + self.width, self.p_y, self.y + self.width]
        z3 = [self.z, self.z + self.height, self.z]
        ax.plot_trisurf(x3, y3, z3)

        #surface 4
        x4 = [self.x + self.length, self.p_x, self.x + self.length]
        y4 = [self.y, self.p_y, self.y + self.width]
        z4 = [self.z, self.z + self.height, self.z]
        ax.plot_trisurf(x4, y4, z4)

        #surface 5
        x5 = [self.x, self.p_x, self.x + self.length]
        y5 = [self.y, self.p_y, self.y]
        z5 = [self.z, self.z + self.height, self.z]
        ax.plot_trisurf(x5, y5, z5)

        plt.show()


class Frustum():
    def __init__(self, x, y, z, length, width, height, p_x, p_y, p_length, p_width):
        self.x = x
        self.y = y
        self.z = z
        self.length = length
        self.width = width
        self.height = height
        self.p_x = p_x
        self.p_y = p_y
        self.p_length = p_length
        self.p_width = p_width

    def reconstruction(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection = "3d")

        xx = np.linspace(self.x, self.x + self.length, 2)
        yy = np.linspace(self.y, self.y + self.width, 2)

        #surface 1
        xx1, yy1 = np.meshgrid(xx, yy)
        ax.plot_surface(xx1, yy1, np.full_like(xx1, self.z))

        xx = np.linspace(self.p_x, self.p_x + self.p_length, 2)
        yy = np.linspace(self.p_y, self.p_y + self.p_width, 2)

        #surface 2
        xx2, yy2 = np.meshgrid(xx, yy)
        ax.plot_surface(xx2, yy2, np.full_like(xx2, self.height))

        #surface 3
        x3 = [self.x, self.p_x, self.p_x]
        y3 = [self.y, self.p_y, self.p_y + self.p_width]
        z3 = [self.z, self.height, self.height]
        ax.plot_trisurf(x3, y3, z3)
        x3 = [self.x, self.p_x, self.x]
        y3 = [self.y, self.p_y + self.p_width, self.y + self.width]
        z3 = [self.z, self.height, self.z]
        ax.plot_trisurf(x3, y3, z3)

        #surface 4
        x4 = [self.x, self.p_x, self.x + self.length]
        y4 = [self.y + self.width, self.p_y + self.p_width, self.y + self.width]
        z4 = [self.z, self.height, self.z]
        ax.plot_trisurf(x4, y4, z4)
        x4 = [self.p_x, self.x + self.length, self.p_x + self.p_length]
        y4 = [self.p_y + self.p_width, self.y + self.width, self.p_y + self.p_width]
        z4 = [self.height, self.z, self.height]
        ax.plot_trisurf(x4, y4, z4)

        #surface 5
        x5 = [self.x + self.length, self.p_x + self.p_length, self.x + self.length]
        y5 = [self.y + self.width, self.p_y + self.p_width, self.y]
        z5 = [self.z, self.height, self.z]
        ax.plot_trisurf(x5, y5, z5)
        x5 = [self.p_x + self.p_length, self.x + self.length, self.p_x + self.p_length]
        y5 = [self.p_y + self.p_width, self.y, self.p_y]
        z5 = [self.height, self.z, self.height]
        ax.plot_trisurf(x5, y5, z5)

        #surface 6
        x6 = [self.x, self.x + self.length, self.p_x + self.p_length]
        y6 = [self.y, self.y, self.p_y]
        z6 = [self.z, self.z, self.height]
        ax.plot_trisurf(x6, y6, z6)
        x6 = [self.x, self.p_x + self.p_length, self.p_x]
        y6 = [self.y, self.p_y, self.p_y]
        z6 = [self.z, self.height, self.height]
        ax.plot_trisurf(x6, y6, z6)

        plt.show()