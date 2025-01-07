import numpy as np
import cvxopt as opt

def svm_hard_linear(X, Y):
    m, d = X.shape  # Number of samples (m) and dimensions (d)
    
    # Construct the matrix A
    A = np.hstack((np.ones((m, 1)), X))  # Add a column of ones for the bias term
    A = A * Y[:, np.newaxis]  # Multiply each row by its corresponding label

    # Construct the matrices for the quadratic programming problem
    P = opt.matrix(np.diag([0] + [1] * d), tc='d')  # Diagonal matrix [0, 1, 1, ...]
    q = opt.matrix([0] * (d + 1), tc='d')  # Zero vector
    G = opt.matrix(-A, tc='d')  # Negative A for inequality constraints
    h = opt.matrix(-np.ones(m), tc='d')  # -1 for inequality constraints

    # Solve the quadratic programming problem
    sol = opt.solvers.qp(P, q, G, h)
    w = np.array(sol['x']).flatten()  # Extract the solution vector

    # Separate bias term and weights
    b = w[0]  # Bias term
    w = w[1:]  # Weight vector

    return b, w

# Input: Transformed points and labels
X = np.array([
    [1, -2],   # phi_1, phi_2 for x1
    [4, -5],   # phi_1, phi_2 for x2
    [4, -1],   # phi_1, phi_2 for x3
    [5, -2],   # phi_1, phi_2 for x4
    [7, -7],   # phi_1, phi_2 for x5
    [7, 1],    # phi_1, phi_2 for x6
    [7, 1]     # phi_1, phi_2 for x7
])

Y = np.array([-1, -1, -1, 1, 1, 1, 1])  # Labels for the dataset

# Solve for b and w
b, w = svm_hard_linear(X, Y)

# Display the results
print(f"Bias (b): {b}")
print(f"Weights (w): {w}")