from submission import *
from mpl_toolkits.mplot3d import Axes3D

'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
im1 = cv2.imread('../data/im1.png')
im2 = cv2.imread('../data/im2.png')

templeCoords = np.load('../data/templeCoords.npz')
x1s = templeCoords['x1']
y1s = templeCoords['y1']

nums = len(x1s)

q2_1 = np.load('../data/q2_1.npz')
F = q2_1['arr_0'][0]

M1 = np.identity(3)
pad = np.zeros((3, 1))
M1 = np.concatenate((M1, pad), axis=1)

intrinsics = np.load('../data/intrinsics.npz')
K1 = intrinsics['K1']

C1 = np.dot(K1, M1)

q3_3 = np.load('../data/q3_3.npz')
M2 = q3_3['arr_0'][0]
C2 = q3_3['arr_0'][1]

q4_2 = {}
q4_2['F'] = F
q4_2['M1'] = M1
q4_2['M2'] = M2
q4_2['C1'] = C1
q4_2['C2'] = C2

np.savez("../data/q4_2.npz", q4_2)

p1s = []
p2s = []

# get all 2D points on image 2 and get all the 3D points
for i in range(nums):
    if i % 10 == 0:
        print(i)
    p1s.append([x1s[i][0], y1s[i][0]])
    x2, y2 = epipolarCorrespondence(im1, im2, F, x1s[i][0], y1s[i][0])
    p2s.append([x2, y2])

p1s = np.array(p1s)
p2s = np.array(p2s)

ps, _ = triangulate(C1, p1s, C2, p2s)

xs = ps[:, 0]
ys = ps[:, 1]
zs = ps[:, 2]

fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

ax.scatter(xs, ys, zs)
plt.show()
