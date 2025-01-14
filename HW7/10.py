from liblinear.liblinearutil import *
from itertools import combinations_with_replacement
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

def read_linear_format(file_path):
    X, y = [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] == '2' or parts[0] == '6':
                y.append(int(parts[0]))  
                features = {}
                for item in parts[1:]:
                    index, value = item.split(":")
                    features[int(index)] = float(value)
                X.append(features)
    return X, y

X_train, y_train = read_linear_format('train.txt')
X_test, y_test = read_linear_format('test.txt')

# decision stump on the i-th dimension
def decision_stump(x, s, i, theta):
    if x[i-1] - theta >= 0:
        return s
    else:
        return -s

def ZeroOneError(prediction, y):
    if prediction == y:
        return 0
    else:
        return 1

def regularized_E_in(u, predictions, y_data):
    error_sum = 0
    for i in range(len(predictions)):
        individual_error = ZeroOneError(predictions[i], y_data[i])
        weighted_individual_error = individual_error * u[i]
        error_sum += weighted_individual_error
    avg_error = error_sum / len(predictions)
    return avg_error

# iteration number T
T = 500
