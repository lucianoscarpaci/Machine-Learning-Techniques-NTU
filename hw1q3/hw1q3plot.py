import numpy as np
import matplotlib.pyplot as plt

# Kernel function (hw1q3 kernel)
def hw1q3_kernel(x1, x2):
    return (1 + np.dot(x1, x2)) ** 2

# Dataset (X: input points, Y: labels)
X = np.array([
    [1, 0],    # Example points
    [0, 1],
    [0, -1],
    [-1, 0],
    [0, 2],
    [0, -2],
    [-2, 0]
])
Y = np.array([-1, -1, -1, 1, 1, 1, 1])  # Labels

# Example alpha values and bias (from solution)
alpha = np.array([6.22247107e-10, 4.85992056e-3, 1.51515106e-2, 2.00114298e-2,
                  9.42767506e-10, 5.52238284e-10, 5.52238284e-10])
b = -29.695823266443394  # Bias term

# Plot the data points
plt.figure(figsize=(8, 6))
plt.title('SVM Decision Boundary with hw1q3 Kernel')
plt.plot(X[Y == 1, 0], X[Y == 1, 1], 'ro', label='+1 examples')
plt.plot(X[Y == -1, 0], X[Y == -1, 1], 'bx', label='-1 examples')

# Grid setup
beg, end, npoints = -3, 3, 100
xrange = np.linspace(beg, end, npoints)
yrange = np.linspace(beg, end, npoints)
xgrid, ygrid = np.meshgrid(xrange, yrange)
zgrid = np.empty((npoints, npoints))

# Compute the decision function over the grid
for i in range(npoints):
    for j in range(npoints):
        pos = np.array([xgrid[i, j], ygrid[i, j]])
        zgrid[i, j] = b + np.sum([alpha[n] * Y[n] * hw1q3_kernel(X[n], pos) for n in range(len(X))])


# Debug the Z-grid sample values
print("Z-grid sample values:")
print(zgrid[:5, :5])
#Z-grid sample values:                                                                                                                                                           
#[[-29.63750417 -29.64713317 -29.65661518 -29.66595017 -29.67513815]
# [-29.62905332 -29.63868233 -29.64816433 -29.65749932 -29.66668731]
# [-29.62074949 -29.6303785  -29.6398605  -29.64919549 -29.65838347]
# [-29.61259266 -29.62222167 -29.63170367 -29.64103866 -29.65022665]
# [-29.60458284 -29.61421185 -29.62369385 -29.63302884 -29.64221682]]

# Plot the decision boundary (f(x) = 0)
#plt.contour(xgrid, ygrid, zgrid, levels=[0], colors='green', linewidths=2)
plt.contour(xgrid, ygrid, zgrid, levels=np.linspace(-29.7, -29.6, 11), colors='green', linewidths=1)
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
#plt.savefig('hw1q3_plot.png')