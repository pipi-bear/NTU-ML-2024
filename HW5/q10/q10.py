from liblinear.liblinearutil import *
from itertools import combinations_with_replacement
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

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

def run_single_experiment(experiment):
    np.random.seed(experiment)
    min_Ein = np.inf
    opt_log10_lambda = 0
    
    for log10_lambda in (-2, -1, 0, 1, 2, 3):
        c = 1 / (10 ** log10_lambda)
        prob = problem(y_train, X_train)
        param = parameter('-s 6 -c ' + str(c))
        model = train(prob, param)

        train_label, _, _ = predict(y_train, X_train, model)
        Ein = ZeroOneError(train_label, y_train)
        if Ein == min_Ein:
            opt_log10_lambda = max(opt_log10_lambda, log10_lambda)
            if opt_log10_lambda == log10_lambda:
                opt_model = model
        elif Ein < min_Ein:
            min_Ein = Ein
            opt_log10_lambda = log10_lambda
            opt_model = model

    test_label, _, _ = predict(y_test, X_test, opt_model)
    Eout = ZeroOneError(test_label, y_test)
    
    W = np.array(opt_model.get_decfun()[0])
    non_zero_count = np.count_nonzero(W)
    
    return Eout, non_zero_count

# aim: Run experiments in parallel
experiment_amount = 1126
n_cores = multiprocessing.cpu_count() - 1
results = Parallel(n_jobs=n_cores)(                                                     
    delayed(run_single_experiment)(i) for i in tqdm(range(experiment_amount))
)

Eouts, non_zero_count_list = zip(*results)
Eouts = list(Eouts)
non_zero_count_list = list(non_zero_count_list)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# aim: plotting
# subaim: First subplot (Eout values)
ax1.hist(Eouts, bins=30)
ax1.set_title('Eout values')
ax1.set_xlabel('Eout')
ax1.set_ylabel('Frequency')

# subaim: Second subplot (non-zero components)
min_val = int(min(non_zero_count_list))
max_val = int(max(non_zero_count_list))
integer_bins = np.arange(min_val, max_val + 2) - 0.5  # +2 to include max_val, -0.5 for bin edges

ax2.hist(non_zero_count_list, 
         bins=integer_bins,
         align='mid',
         rwidth=0.8)  

ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax2.set_title('Amount of non-zero components in g')
ax2.set_xlabel('Amount of non-zero components')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# aim: print some additional information about the result statistics
print("\nStatistics for Eout:")
print(f"Mean: {np.mean(Eouts):.4f}")
print(f"Median: {np.median(Eouts):.4f}")
print(f"Standard Deviation: {np.std(Eouts):.4f}")
print(f"Min: {np.min(Eouts):.4f}")
print(f"Max: {np.max(Eouts):.4f}")

print("\nStatistics for Non-zero Components:")
print(f"Mean: {np.mean(non_zero_count_list):.1f}")
print(f"Median: {np.median(non_zero_count_list):.1f}")
print(f"Standard Deviation: {np.std(non_zero_count_list):.1f}")
print(f"Min: {np.min(non_zero_count_list)}")
print(f"Max: {np.max(non_zero_count_list)}")