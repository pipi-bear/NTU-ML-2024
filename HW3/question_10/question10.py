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

# explain: the data X returned is in the form of a list of dictionaries, where each dictionary contains the features of a data point
# ex: X[0] = {1: -0.993496, 2: -0.993043, 3: -0.850291, 4: -0.963479, 5: -0.960727, 6: -0.900596, 7: -0.96642, 8: -0.863996, 9: -0.606175, 10: -0.999291, 11: 0.0811894, 12: 0.651101}

# explain: the label y is a list of floats
# ex: y[0] = 90.0

N = 32
feature_amount = 12
in_out_sample_error = []

# aim: perform linear regression for 1126 times, calculate the in-sample error and estimate the out of sample error
for experiment in tqdm(range(1)):       # note: range(1) is for testing, need to be modified
    np.random.seed(experiment)

    random_sample_indices = np.random.choice(len(X), N, replace=False)
    X_sample = [X[i] for i in random_sample_indices]
    y_sample = [y[i] for i in random_sample_indices]

    # subaim: convert each sample data points (which is a dictionary) to a ndarray and save in X_sample_array
    X_sample_array = []
    for sample in X_sample:
        input_vector = np.concatenate((np.array([1]), np.zeros(feature_amount)))
        # explain: originally we need subtract 1 because indices in libsvm format start from 1, but now we add the 0-th element (1) to each input vector
        for index, value in sample.items():
            input_vector[index] = value        
        X_sample_array.append(input_vector)
    
    # subaim: convert X_sample_array to a matrix, and y_sample to an array
    X_mat = np.array(X_sample_array)
    y_array = np.array(y_sample)

    # subaim: perform linear regression
    # explain: we do linear regression by using the normal equation (w = (X^TX)^(-1)X^Ty)
    # explain: we use np.linalg.inv() calculate the inverse of X^TX, and use @ to denote matrix multiplication
    w = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_array

    # subaim: calculate the in-sample error
    # explain: the in-sample error is the mean squared error of the N points
    in_sample_error = np.mean((X_mat @ w - y_array) ** 2)

    # subaim: estimate the out of sample error
    # explain: the out of sample error is estimated by averaging the squared error of the rest of the 8192 - N data points
    out_of_sample_error = 0
    for i in range(len(X)):
        if i not in random_sample_indices:
            unseen_input_vector = np.concatenate((np.array([1]), np.zeros(feature_amount))) 
            for index, value in X[i].items():
                unseen_input_vector[index] = value
            out_of_sample_error += (unseen_input_vector @ w - y[i]) ** 2
    out_of_sample_error /= (len(X) - N)
    in_out_sample_error.append((in_sample_error, out_of_sample_error))

# aim: plot the in-sample error and out of sample error for each experiment as a scatter plot
plt.figure(figsize=(10, 8)) 
plt.scatter([i[0] for i in in_out_sample_error], [i[1] for i in in_out_sample_error], 
            s=20, alpha=0.5)  
plt.xlabel('In-sample error')
plt.ylabel('Out-of-sample error', labelpad=10)  
plt.grid(alpha=0.6)  
plt.show()