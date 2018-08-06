import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np

X =[]
Y =[]

# We load data
for line in open("../files/regression/data_2d.csv"):
	ar = line.split(',')
	# 1 is the bias term added. For y = w0 + w1x1 + w2x2 + ... + wDxD
	X.append([1, float(ar[0]), float(ar[1])])
	Y.append(float(ar[2]))

X = np.array(X)
Y = np.array(Y)

# Plotting the data

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

plt.show()

# Calculating weight
W = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
# We write the code as Y = Wtranspose X. 
# But here X is an NXD matrix where each sample is row. It is easier to do XW
Yhat = np.dot(X,W)

# Compute R squared.
SSres = Y-Yhat
SStot = Y-Y.mean()

r_sq = 1 - SSres.dot(SSres) / (SStot.dot(SStot))

print "r_squared =", r_sq 