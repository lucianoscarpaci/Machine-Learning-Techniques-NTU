# Find the y points

#import numpy as np
# Code the transformed points and their labels
# original x = (x1, x2) y          transformed z = (phi1(x), phi2(x))
# (1, 0)                -1              (1, -2)
def phi1(x):
	x1, x2 = x
	return x2**2 - 2*x1 + 3

def phi2(x):
	x1, x2 = x
	return x1**2 - 2 * x2 - 3

data_points = [
    (1, 0),
    (0, 1),
    (0, -1),
    (-1, 0),
    (0, 2),
    (0, -2),
    (-2, 0)
]

transformed_points = [[phi1(x), phi2(x)] for x in data_points]
for i, (original, transformed) in enumerate(zip(data_points, transformed_points)):
    print(f"x_{i}: {original}, phi1 = {transformed[0]}, phi2 = {transformed[1]}")