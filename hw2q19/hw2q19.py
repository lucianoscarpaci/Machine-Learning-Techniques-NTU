import numpy as np
import matplotlib.pyplot as plt

def load_file(file_name):
    lines = open(file_name, 'r').readlines()
    return np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])

def solve_equation(K, lamb, y):
    return np.linalg.solve(K + np.eye(K.shape[0]) * lamb, y)

def kernel_matricize(X, kernel):
	m = X.shape[0]
	mat = np.zeros((m, m))
	for i in range(m):
		for j in range(i + 1):
			mat[i, j] = kernel(X[i], X[j])
	for i in range(m):
		for j in range(i + 1, m):
			mat[i, j] = mat[j, i]
	return mat

def rbf_kernel(x1, x2, gamma):
	x = x1 - x2
	return np.exp(-gamma * np.dot(x, x))

def predict(x, X, beta, kernel):
    dots = np.array([kernel(x, v) for v in X])
    s = np.sum(dots * beta)
    return 2 * int(s > 0) - 1

def hw2q19():
	train = load_file('hw2_lssvm_all.dat')

	m_train = 400
	X_train = train[:m_train, :-1]
	Y_train = train[:m_train, -1].astype(int)

	m_test = 200
	X_test = train[m_train:m_train + m_test, :-1]
	Y_test = train[m_train:m_train + m_test, -1].astype(int)

	E_in_values = []
	E_out_values = []

	for gamma in [32, 2, 0.125]:
		for lamb in [0.001, 1, 1000]:

			K = kernel_matricize(X_train, lambda x1, x2: rbf_kernel(x1, x2, gamma))
			beta = solve_equation(K, lamb, Y_train)

			E_in = np.mean([predict(x, X_train, beta, lambda x1, x2: rbf_kernel(x1, x2, gamma)) != y for x, y in zip(X_train, Y_train)])
			E_out = np.mean([predict(x, X_train, beta, lambda x1, x2: rbf_kernel(x1, x2, gamma)) != y for x, y in zip(X_test, Y_test)])

			E_in_values.append(E_in)
			E_out_values.append(E_out)

	plt.plot(range(1, len(E_in_values) + 1), E_in_values, label='E_in')
	plt.plot(range(1, len(E_out_values) + 1), E_out_values, label='E_out')
	plt.xlabel('t')
	plt.ylabel('In/Out Error')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	hw2q19()