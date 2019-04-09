import cv2
import numpy as np
from numpy.linalg import svd, eig
import helper as hl
from sympy import *
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, shift

"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    length = len(pts1)

    pts1 = np.array(pts1, dtype=float)
    pts2 = np.array(pts2, dtype=float)

    pts1 /= M
    pts2 /= M

    x1 = pts1[:, 0].T
    y1 = pts1[:, 1].T
    x2 = pts2[:, 0].T
    y2 = pts2[:, 1].T
    ones = np.ones(length).T

    U = np.vstack([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, ones]).T
    _, _, V = svd(U)
    F = V.T[:, -1]
    F = F.reshape(3, 3)

    F = hl._singularize(F)
    F = hl.refineF(F, pts1, pts2)
    unscale = np.diag((1 / M, 1 / M, 1))
    F = np.dot(np.transpose(unscale), F)
    F = np.dot(F, unscale)

    return F


'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    print(pts1.shape)
    # Replace pass by your implementation
    length = len(pts1)

    pts1 = np.array(pts1, dtype=float)
    pts2 = np.array(pts2, dtype=float)
    pts1 /= M
    pts2 /= M

    x1 = pts1[:, 0].T
    y1 = pts1[:, 1].T
    x2 = pts2[:, 0].T
    y2 = pts2[:, 1].T
    ones = np.ones(length).T

    U = np.vstack([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, ones]).T

    _, _, V = svd(U)

    F1 = V.T[:, -1]
    F1 = F1.reshape(3, 3)
    F1 = np.transpose(F1)

    F2 = V.T[:, -2]
    F2 = F2.reshape(3, 3)
    F2 = np.transpose(F2)

    u1, s1, v1 = svd(F1)
    s1[2] = 0
    s1 = np.diag(s1)
    F1 = np.dot(u1, s1)
    F1 = np.dot(F1, np.transpose(v1))

    u2, s2, v2 = svd(F2)
    s2[2] = 0
    s2 = np.diag(s2)
    F2 = np.dot(u2, s2)
    F2 = np.dot(F2, np.transpose(v2))

    k = symbols('k')
    k = solve((k*F1[0][0]+(1-k)*F2[0][0])*(k*F1[1][1]+(1-k)*F2[1][1])*(k*F1[2][2]+(1-k)*F2[2][2])
              +(k*F1[0][1]+(1-k)*F2[0][1])*(k*F1[1][2]+(1-k)*F2[1][2])*(k*F1[2][0]+(1-k)*F2[2][0])
              +(k*F1[0][2]+(1-k)*F2[0][2])*(k*F1[1][0]+(1-k)*F2[1][0])*(k*F1[2][1]+(1-k)*F2[2][1])
              -(k*F1[2][0]+(1-k)*F2[2][0])*(k*F1[1][1]+(1-k)*F2[1][1])*(k*F1[0][2]+(1-k)*F2[0][2])
              -(k*F1[1][0]+(1-k)*F2[1][0])*(k*F1[0][1]+(1-k)*F2[0][1])*(k*F1[2][2]+(1-k)*F2[2][2])
              -(k*F1[0][0]+(1-k)*F2[0][0])*(k*F1[2][1]+(1-k)*F2[2][1])*(k*F1[1][2]+(1-k)*F2[1][2]))


    Fs = []
    unscale = np.diag((1 / M, 1 / M, 1))

    for i in range(len(k)):
        param = re(k[i])
        F = param * F1 + (1 - param) * F2
        F = np.array(F, dtype=float)

        F = hl.refineF(F, pts1, pts2)
        F = np.dot(np.transpose(unscale), F)
        F = np.dot(F, unscale)
        Fs.append(F)

    return Fs


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = np.dot(np.transpose(K2), F)
    E = np.dot(E, K1)

    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    length = len(pts1)
    P = np.zeros((length, 4))
    pts1 = np.concatenate((pts1, np.ones((length, 1))), axis=1)
    pts2 = np.concatenate((pts2, np.ones((length, 1))), axis=1)

    for i in range(length):
        A = np.zeros((4, 4))
        A[0] = pts1[i][0] * C1[2, :] - C1[0, :]
        A[1] = pts1[i][1] * C1[2, :] - C1[1, :]
        A[2] = pts2[i][0] * C2[2, :] - C2[0, :]
        A[3] = pts2[i][1] * C2[2, :] - C2[1, :]

        u, s, v = svd(A)
        last_ev = np.transpose(v.T[:, -1])
        last_ev /= last_ev[3]
        P[i] = last_ev

    P = np.transpose(P)
    p1 = np.dot(C1, P)
    p2 = np.dot(C2, P)

    p1 = np.transpose(p1)
    p2 = np.transpose(p2)

    for i in range(length):
        p1[i] /= p1[i][2]
        p2[i] /= p2[i][2]

    err1 = p1 - pts1
    err2 = p2 - pts2

    err = np.sum(err1**2 + err2**2)
    P = P[:3]
    P = P.T
    return P, err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    p1 = np.array([x1, y1, 1])
    e_line = np.dot(F, p1)
    e_line /= np.sqrt(np.sum(e_line ** 2))

    radius = 30

    # get small area from im1
    area1 = im1[y1-radius:y1+radius, x1-radius:x1+radius, :]

    # x should be within 480, y should be within 640
    x_lim = im1.shape[1]
    y_lim = im1.shape[0]

    cnt = 0
    min_diff = 999999
    x2_pred = -1
    y2_pred = -1
    # iterate through im2
    for y2 in range(y_lim):
        x2 = -(y2 * e_line[1] + e_line[2]) / e_line[0]
        x2 = int(round(x2))
        # get the area from im2
        if radius <= x2 < x_lim - radius and radius <= y2 < y_lim - radius:
            area2 = im2[y2 - radius:y2 + radius, x2 - radius:x2 + radius, :]
            cnt += 1
            diff = area_diff(area1, area2)
            if diff < min_diff:
                min_diff = diff
                x2_pred = x2
                y2_pred = y2

    return x2_pred, y2_pred


# calculate the diff between two areas with the same shape that have 3 channels
def area_diff(area1, area2):
    area1 = gaussian_filter(area1, sigma=3)
    area2 = gaussian_filter(area2, sigma=3)
    diff = area1 - area2
    diff = diff ** 2
    diff = np.sum(diff)

    return diff


if __name__ == '__main__':
    im1 = cv2.imread('../data/im1.png')
    im2 = cv2.imread('../data/im2.png')

    w = im1.shape[0]
    h = im1.shape[1]
    M = max(w, h)

    some_corresp = np.load('../data/some_corresp.npz')
    pts1 = some_corresp['pts1']
    pts2 = some_corresp['pts2']

    # Q2.1
    F = eightpoint(pts1, pts2, M)
    print(F)
    hl.displayEpipolarF(im1, im2, F)
    q2_1 = (F, M)
    np.savez("../data/q2_1.npz", q2_1)

    # Q2.2
    start = 1
    pts1 = pts1[start:start+7]
    pts2 = pts2[start:start+7]
    Fs = sevenpoint(pts1, pts2, M)

    for F in Fs:
        print(F)

    for F in Fs:
        hl.displayEpipolarF(im1, im2, F)

    q2_2 = (Fs, M)
    np.savez("../data/q2_2.npz", q2_2)

    # Q3.1
    intrinsics = np.load('../data/intrinsics.npz')
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']

    q2_1 = np.load('../data/q2_1.npz')
    F = q2_1['arr_0'][0]

    E = essentialMatrix(F, K1, K2)
    print(E)

    # Q4.1
    q2_1 = np.load('../data/q2_1.npz')
    F = q2_1['arr_0'][0]

    pt1s, pt2s = hl.epipolarMatchGUI(im1, im2, F)

    q4_1 = (F, pt1s, pt2s)
    np.savez("../data/q4_1.npz", q4_1)
