import numpy as np

'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

templeCoords = np.load('../data/templeCoords.npz')
x1s = templeCoords['x1']
y1s = templeCoords['y1']

num = len(x1s)



print(num)
