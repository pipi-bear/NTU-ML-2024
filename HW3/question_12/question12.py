import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def read_libsvm_format(file_path):
    X, y = [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            y.append(float(parts[0])) 
            features = {}
            for item in parts[1:]:
                index, value = item.split(":")
                features[int(index)] = float(value)
            X.append(features)
    return X, y

# aim: read the data from the file 'cpusmall_scale'
X, y = read_libsvm_format('cpusmall_scale')

N_list = list(range(25, 2025, 25))
seed = 0

# aim: for each sample size N, perform linear regression for 16 times, calculate the average in-sample error and estimate the average out of sample error
avg_in_error_for_each_N = []
avg_out_error_for_each_N = []
for N in tqdm(N_list):
    in_out_sample_error = []
    # subaim: perform linear regression for 16 times, calculate the in-sample error and estimate the out of sample error
    for experiment in range(16):
        seed += 1
        np.random.seed(seed)

        random_sample_indices = np.random.choice(len(X), N, replace=False)
        X_sample = [X[i] for i in random_sample_indices]
        y_sample = [y[i] for i in random_sample_indices]

        # subsubaim: convert each sample data points (which is a dictionary) to a ndarray and save in X_sample_array
        X_sample_array = []
        for sample in X_sample:
            input_vector = [1, sample.get(1, 0), sample.get(2, 0)]
            X_sample_array.append(input_vector)
        
        # subsubaim: convert X_sample_array to a matrix, and y_sample to an array
        X_mat = np.array(X_sample_array)
        y_array = np.array(y_sample)

        # subsubaim: perform linear regression
        w = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_array

        # subsubaim: calculate the in-sample error
        in_sample_error = np.mean((X_mat @ w - y_array) ** 2)

        # subsubaim: estimate the out of sample error
        out_sample_indices = list(set(range(len(X))) - set(random_sample_indices))
        X_out = np.array([[1, X[i].get(1, 0), X[i].get(2, 0)] for i in out_sample_indices])
        y_out = np.array([y[i] for i in out_sample_indices])
        out_of_sample_error = np.mean((X_out @ w - y_out) ** 2)
        in_out_sample_error.append((in_sample_error, out_of_sample_error))
    
    # subaim: calculate the average in-sample error and out of sample error for each N
    avg_in_sample_error = np.mean([error[0] for error in in_out_sample_error])
    avg_out_of_sample_error = np.mean([error[1] for error in in_out_sample_error])

    avg_in_error_for_each_N.append(avg_in_sample_error)
    avg_out_error_for_each_N.append(avg_out_of_sample_error)


# aim: plot E_in(N) and E_out(N)
plt.figure(figsize=(12, 8))

plt.plot(N_list, avg_in_error_for_each_N, label='E_in(N)', color='blue')
plt.plot(N_list, avg_out_error_for_each_N, label='E_out(N)', color='red')
plt.xlabel('N (Sample Size)')
plt.ylabel('Error')
plt.title('In-sample and Out-of-sample Error vs. Sample Size')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()