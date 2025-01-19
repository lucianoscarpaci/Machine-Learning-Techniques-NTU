from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import random

def load_file(file_name):
    lines = open(file_name, 'r').readlines()
    return np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])

def train_random_pruned_dtrees(X_train, Y_train, num_trees=80, max_depth_range=(1, 3)):
    dtrees = []
    for _ in range(num_trees):
        # Randomly select a max_depth for the current tree
        max_depth = random.randint(max_depth_range[0], max_depth_range[1])
        
        # Initialize and train the DecisionTreeClassifier with the selected max_depth
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random.randint(0, 10000))
        clf.fit(X_train, Y_train)
        
        # Append the trained tree to the list
        dtrees.append(clf)
    
    return dtrees

def predict_with_dtrees(dtrees, X):
    # Aggregate predictions from all trees
    predictions = np.zeros((len(dtrees), X.shape[0]))
    for i, tree in enumerate(dtrees):
        predictions[i] = tree.predict(X)
    
    # Majority vote
    final_predictions = np.sign(np.sum(predictions, axis=0))
    return final_predictions

def hw3q19_20():
    train = load_file('hw3_train.dat')
    X_train = train[:, :-1]
    Y_train = train[:, -1].astype(int)
    
    test = load_file('hw3_test.dat')
    X_test = test[:, :-1]
    Y_test = test[:, -1].astype(int)

    # Lists to store the train and test errors
    train_errors = []
    test_errors = []

    # Fit the model with progress bar
    for _ in tqdm(range(100), desc="Training Random Pruned Decision Trees"):
        # Train random pruned decision trees
        dtrees = train_random_pruned_dtrees(X_train, Y_train, num_trees=80, max_depth_range=(1, 3))

        # Predict and calculate errors
        train_predictions = predict_with_dtrees(dtrees, X_train)
        test_predictions = predict_with_dtrees(dtrees, X_test)

        train_error = 1 - accuracy_score(Y_train, train_predictions)
        test_error = 1 - accuracy_score(Y_test, test_predictions)
        
        # Append the training errors and testing errors
        train_errors.append(train_error)
        test_errors.append(test_error)
        
    print('----------------------------------------')
    print('         Homework 3 Question 19         ')
    print('----------------------------------------')
    print('avg(E_in) = %f' % np.mean(train_errors))
    print()

    print('----------------------------------------')
    print('         Homework 3 Question 20         ')
    print('----------------------------------------')
    print('avg(E_out) = %f' % np.mean(test_errors))
    print()

hw3q19_20()