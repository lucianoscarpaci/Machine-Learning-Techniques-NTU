from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt

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

def hw1q15():
    # Create a LinearSVC model with C=0.01, linear kernel and no shrinking
    model = LinearSVC(C=0.01, dual=False)
    # Extracts the rows where the label is 0
    X_train_0 = X_train
    # Converts the labels to 0 or 1
    Y_train_0 = (Y_train == 0).astype(int)
    # Fit the model
    model.fit(X_train_0, Y_train_0)
    # Extract the weights
    w = model.coef_[0]
    # Extract the intercept
    b = model.intercept_[0]
    # Print the weights, norm of the weights and the intercept
    print(f"w = {w}")
    print(f"norm(w) = {np.linalg.norm(w, ord=2)}")
    print(f"b = {b}")

hw1q15()
