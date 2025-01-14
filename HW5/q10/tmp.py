from liblinear.liblinearutil import *
from itertools import combinations_with_replacement
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from tqdm import tqdm

train_data = 'mnist.scale'
test_data = 'mnist.scale.t'

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

X_train, y_train = read_linear_format(train_data)
X_test, y_test = read_linear_format(test_data)

def ZeroOneError(predictions, y):
    return sum(p_i != y_i for p_i, y_i in zip(predictions, y)) / len(predictions)

Eouts = []
non_zero_count_list = []

for experiment in tqdm(range(5)):
    min_Ein = np.inf
    opt_log10_lambda = 0
    seed = np.random.seed(experiment)
    for log10_lambda in (-2, -1, 0, 1, 2, 3):
        train_pred_res = []
        c = 1 / (10 ** log10_lambda)
        prob = problem(y_train, X_train)
        param = parameter('-s 6 -c ' + str(c))
        model = train(prob, param)

        train_label, _, _ = predict(y_train, X_train, model)
        Ein = ZeroOneError(train_label, y_train)
        print(f'Ein for experiment {experiment} is {Ein}')
        if Ein == min_Ein:
            opt_log10_lambda = max(opt_log10_lambda, log10_lambda)      # break tie by choosing the larger lambda
            if opt_log10_lambda == log10_lambda:
                opt_model = model
        elif Ein < min_Ein:
            minEin = Ein
            opt_log10_lambda = log10_lambda
            opt_model = model

    #print('The best log_10(λ*) = ', opt_log10_lambda)
    c_test  = 1 / (10 ** opt_log10_lambda)
    prob_test = problem(y_test, X_test)
    param_test = parameter('-s 6 -c ' + str(c_test))
    model_test = opt_model

    test_label, _, _ = predict(y_test, X_test, model_test)
    Eout = ZeroOneError(test_label, y_test)
    
    Eouts.append(Eout)
    
    W = model_test.get_decfun()[0]
    non_zero_count = np.count_nonzero(W)
    non_zero_count_list.append(non_zero_count)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(Eouts, bins='auto', alpha=0.7)
ax1.set_title('Eout values')
ax1.set_xlabel('Eout')
ax1.set_ylabel('Frequency')

ax2.hist(non_zero_count_list, bins='auto', alpha=0.7)
ax2.set_title('Amount of non-zero components in g')
ax2.set_xlabel('Amount of non-zero components')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

