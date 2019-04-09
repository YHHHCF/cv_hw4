import numpy as np
import cv2
from submission import essentialMatrix, triangulate
import helper as hl

'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

im1 = cv2.imread('../data/im1.png')
im2 = cv2.imread('../data/im2.png')

w = im1.shape[0]
h = im1.shape[1]
M = max(w, h)

some_corresp = np.load('../data/some_corresp.npz')
pts1 = some_corresp['pts1']
pts2 = some_corresp['pts2']

intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']
K2 = intrinsics['K2']

q2_1 = np.load('../data/q2_1.npz')
F = q2_1['arr_0'][0]

E = essentialMatrix(F, K1, K2)

M2s = hl.camera2(E)

M1 = np.identity(3)
pad = np.zeros((3, 1))
M1 = np.concatenate((M1, pad), axis=1)

C1 = np.dot(K1, M1)

err_min = 999999
for i in range(M2s.shape[2]):
    M2 = M2s[:, :, i]
    C2 = np.dot(K2, M2)
    P, err = triangulate(C1, pts1, C2, pts2)

    Z = P[:, 2]
    # print(Z)
    positive = (Z > 0).all()
    if positive:
        print("correct one")
        print("err", err)
        q3_3 = (M2, C2, P)
        np.savez("../data/q3_3.npz", q3_3)
