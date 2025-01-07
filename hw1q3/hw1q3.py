import numpy as np
import cvxopt as opt

# define the custom kernel function
def custom_kernel(x1, x2):
	return (1 + np.dot(x1, x2)) ** 2

# define the svm hard kernel function

# SVM with kernel implementation
def svm_hard_kernel(X, Y, kernel):
    m, d = X.shape  # Number of samples (m) and dimensions (d)
    
    # Initialize kernel matrix
    K = np.zeros((m, m))
    
    # Construct the kernel matrix: K[i, j] = Y[i] * Y[j] * K(X[i], X[j])
    for i in range(m):
        for j in range(m):
            K[i, j] = Y[i] * Y[j] * kernel(X[i], X[j])
    
    # 1/2 x'Px + q'x
    P = opt.matrix(K, tc='d')  # Kernel matrix
    q = opt.matrix(-np.ones(m), tc='d')  # -1 vector

    # Ax = b (equality constraint for hard margin SVM)
    A = opt.matrix(Y, (1, m), tc='d')  # Y transposed
    b = opt.matrix(0.0, tc='d')  # Scalar 0

    # Gx <= h (inequality constraint for alpha >= 0)
    G = opt.matrix(-np.identity(m), tc='d')  # Negative identity matrix
    h = opt.matrix(0.0, (m, 1), tc='d')  # Zero vector

    # Solve the quadratic programming problem
    sol = opt.solvers.qp(P, q, G, h, A, b)

    # Extract Lagrange multipliers (alphas)
    alpha = np.array(sol['x']).flatten()

    # Find a support vector index (alpha > 0)
    s = np.argmax(alpha)  # Get the index of the first support vector
    
    # Compute the bias term b
    bias = Y[s] - np.sum(alpha * Y * K[:, s])

    return {'alpha': alpha, 'bias': bias}

# Dataset X = input points and Y = labels
X = np.array([
	[1, -2],
	[4, -5],
	[4, -1],
	[5, -2],
	[7, -7],
	[7, 1],
	[7, 1]
])

Y = np.array([-1, -1, -1, 1, 1, 1, 1])

result = svm_hard_kernel(X, Y, kernel=custom_kernel)

#Output the Alpha values and the bias
print(f"Alphas: {result['alpha']}")
print(f"Bias: {result['bias']}")
# The alpha values are the Lagrange multipliers
# Bias represents the intercept of the decision boundary