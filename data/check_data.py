import numpy as np

intrinsics = np.load('./intrinsics.npz')
print(intrinsics.files)
print(intrinsics['K1'].shape)
print(intrinsics['K2'].shape)

some_corresp = np.load('./some_corresp.npz')
print(some_corresp.files)
print(some_corresp['pts1'].shape)
print(some_corresp['pts2'].shape)

templeCoords = np.load('./templeCoords.npz')
print(templeCoords.files)
print(templeCoords['x1'].shape)
print(templeCoords['y1'].shape)

q2_1 = np.load('./q2_1.npz')
print(q2_1['arr_0'])

q2_2 = np.load('./q2_2.npz')
print(q2_2['arr_0'])

q3_3 = np.load('./q3_3.npz')
print(q3_3['arr_0'][0].shape)
print(q3_3['arr_0'][1].shape)
print(q3_3['arr_0'][2].shape)
