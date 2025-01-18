from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

def load_file(file_name):
    lines = open(file_name, 'r').readlines()
    return np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])

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
    for _ in tqdm(range(100), desc="Training RandomForest"):
        # Initialize the RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=300, random_state=42)
        rf.fit(X_train, Y_train)

        # Predict and calculate errors
        train_predictions = rf.predict(X_train)
        test_predictions = rf.predict(X_test)

        train_error = 1 - accuracy_score(Y_train, train_predictions)
        test_error = 1 - accuracy_score(Y_test, test_predictions)
        # Append the training errors and testing errors
        train_errors.append(train_error)
        test_errors.append(test_error)
        
    print('----------------------------------------')
    print('         Homework 3 Question 19         ')
    print('----------------------------------------')
    print('avg(E_in) = %f' % train_error)
    print()

    print('----------------------------------------')
    print('         Homework 3 Question 20         ')
    print('----------------------------------------')
    print('avg(E_out) = %f' % test_error)
    print()

hw3q19_20()