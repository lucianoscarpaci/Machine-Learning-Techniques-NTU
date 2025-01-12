import numpy as np
import matplotlib.pyplot as plt

def plot_sign(X, Y):
    plt.plot(X[Y > 0, 0], X[Y > 0, 1], 'ro')
    plt.plot(X[Y < 0, 0], X[Y < 0, 1], 'go')

def load_file(file_name):
    lines = open(file_name, 'r').readlines()
    return np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])

def predict(X, s, i, thres):
    return (s * (2 * (X[:, i] > thres) - 1)).astype(int)

def predict_final(X, gs):
    # g_t = (s, i, thres)
    predictions = np.array([predict(X, s, i, thres) * alpha for (s, i, thres, alpha) in gs])
    prediction = np.sum(predictions, 0)
    prediction = (2 * (prediction > 0) - 1).astype(int)
    return prediction

def update_u_alpha(Y_predict, Y, u):

    # correctness array, True for error
    errs = (Y_predict != Y)

    # incorrect rate
    epsilon = float(np.sum(errs * u)) / (np.sum(u))

    coeff_incorrect = np.sqrt((1 - epsilon) / epsilon)
    coeff_correct = np.sqrt(epsilon / (1 - epsilon))

    # the coef's applied to u
    coef = [coeff_incorrect if err else coeff_correct for err in errs]
    
    # (new u, alpha, epsilon)
    return (u * coef, np.log(coeff_incorrect), epsilon)

def hw2q12():
	X = load_file('hw2_adaboost_train.dat')
	Y = X[:, -1]
	X = X[:, :-1]

	X_test = load_file('hw2_adaboost_test.dat')
	Y_test = X_test[:, -1]
	X_test = X_test[:, :-1]

	T = 300
	N = len(Y)
	u = np.ones(N) / N

	gs = []
	for t in range(T):
		min_err = 1
		for i in range(X.shape[1]):
			for thres in np.unique(X[:, i]):
				for s in [1, -1]:
					Y_predict = predict(X, s, i, thres)
					u, alpha, epsilon = update_u_alpha(Y_predict, Y, u)
					if epsilon < min_err:
						min_err = epsilon
						best_g = (s, i, thres, alpha)
		gs.append(best_g)

	E_in = []
	for t in range(1, T + 1):
		Y_predict = predict_final(X, gs[:t])
		E_in_t = np.mean(Y_predict != Y)
		E_in.append(np.mean(Y_predict != Y))
		print(f"E_in for t={t}: {E_in_t}")

	E_out = []
	for t in range(1, T + 1):
		Y_predict = predict_final(X_test, gs[:t])
		E_out.append(np.mean(Y_predict != Y_test))

	plt.plot(range(1, T + 1), E_in, label='E_in')
	plt.plot(range(1, T + 1), E_out, label='E_out')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	hw2q12()