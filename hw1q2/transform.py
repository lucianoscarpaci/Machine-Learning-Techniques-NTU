# Find the y points

#import numpy as np
# Code the transformed points and their labels
# original x = (x1, x2) y          transformed z = (phi1(x), phi2(x))
# (1, 0)                -1              (1, -2)
# Labels are supposedly -1, -1, -1, +1, +1, +1, +1
# solve to verify the labels
def phi1(x):
	x1, x2 = x
	return x2**2 - 2*x1 + 3

def phi2(x):
	x1, x2 = x
	return x1**2 - 2 * x2 - 3

# data points and their labels y
data_points = [
    ((1, 0), -1),
    ((0, 1), -1),
    ((0, -1), -1),
    ((-1, 0), 1),
    ((0, 2), 1),
    ((0, -2), 1),
    ((-2, 0), 1)
]

transformed_points_labels = [[phi1(x), phi2(x), y] for (x, y) in data_points]
# Print the transformed points with their labels
for i, (original, transformed) in enumerate(zip(data_points, transformed_points_labels)):
    print(f"x_{i}: {original}, phi1 = {transformed[0]}, phi2 = {transformed[1]}, label = {transformed[2]}")