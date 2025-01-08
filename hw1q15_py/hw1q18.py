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

