from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

def plot_samples(X, Y):
    plt.plot(X[Y==False, 0], X[Y==False, 1], 'r.')
    plt.plot(X[Y, 0], X[Y, 1], 'g.')

def plot_01(X, Y):
    plt.plot(X[Y == 0, 0], X[Y == 0, 1], 'r.')
    plt.plot(X[Y == 1, 0], X[Y == 1, 1], 'g.')


# Define the Train
train = np.array([np.fromstring(line, dtype=float, sep=' ') for line in open('features.train', 'r').readlines()])
X_train = train[:,1:]
# Extracts the first column and converts to int
Y_train = train[:,0].astype(int)
# Reading and processing testing data
test = np.array([np.fromstring(line, dtype=float, sep=' ') for line in open('features.test', 'r').readlines()])
# Extracts all the columns except the first one from the test array
X_test = test[:,1:]
# Extracts the first column and converts to int
Y_test = test[:,0].astype(int)

# Place holder for plot functions

def hw1q16():
    for idx in (0, 2, 4, 6, 8):
        model = SVC(C=0.01, kernel='poly', degree=2, gamma=1, coef0=1, tol=1e-4, shrinking=True, verbose=False)
        Y_train_i = (Y_train == idx).astype(int)
        model.fit(X_train, Y_train_i)
        Y_predict_i = model.predict(X_train)
        support = model.support_
        coef = model.dual_coef_[0]
        b = model.intercept_[0]
        E_in = np.count_nonzero(Y_train_i != Y_predict_i)
        print('For class %d:' % (idx))
        print('sum(alpha) =', np.sum(np.abs(coef)))
        print('b =', b)
        print('E_in =', E_in)

        fig = plt.figure()
        # plt.suptitle('%d vs rest' % (idx))
        plt.subplot(311)
        plt.title('Training data: green +, red -')
        plot_01(X_train, Y_train_i)
        plt.tick_params(axis='x', labelbottom=False)
        
        plt.subplot(312)
        plt.title('Prediction: green +, red -')
        plot_01(X_train, Y_predict_i)
        plt.tick_params(axis='x', labelbottom=False)

        plt.subplot(313)
        plt.title('Support vectors: blue')
        plt.plot(X_train[:, 0], X_train[:, 1], 'r.')
        plt.plot(X_train[support, 0], X_train[support, 1], 'b.')

    plt.show()

hw1q16()



