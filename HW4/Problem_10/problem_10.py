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

# aim: implement the sigmoid function
def sigmoid(s):
    return 1 / (1 + np.exp(-s))

# aim: implement the sqr error function
# explain: the sqr error function is defined as (y * w^T * x - 1)^2
def sqr_error(w, x, y):
    return (y * w @ x - 1) ** 2


N = 64
in_out_sample_error = []        # this list is used to save the in-sample error and out of sample error for the 1126 experiments while using w_lin

eta = 0.01
w_t = np.zeros(12 + 1)
amount_of_t = 100000 / 200
mult_200_E_in = np.zeros(amount_of_t)   # save the in-sample error of w_t for every t = 200, 400, ..., 100000
mult_200_E_out = np.zeros(amount_of_t)  # save the out of sample error of w_t for every t = 200, 400, ..., 100000

for experiment in tqdm(range(1126)):

    # aim: create the random sample list "X_sample" and "y_sample" for each experiment
    np.random.seed(experiment)

    random_sample_indices = np.random.choice(len(X), N, replace=False)
    X_sample = [X[i] for i in random_sample_indices]
    y_sample = [y[i] for i in random_sample_indices]

    # explain: each X_sample[i] is a dictionary, which looks like the example shown above, and X_sample contains N such dictionaries
    # explain: each y_sample[i] is a float, which is the label of the corresponding data point in X_sample, and y_sample contains N such floats

    # aim: convert each sample data points (which is a dictionary) to a ndarray and save in X_sample_array
    X_sample_array = []
    for sample in X_sample:
        input_vector = np.concatenate((np.array([1]), np.zeros(12)))
        # explain: originally we need to subtract 1 because indices in libsvm format start from 1, but now we add the 0-th element (1) to each input vector
        for index, value in sample.items():
            input_vector[index] = value        
        X_sample_array.append(input_vector)

    # explain: each X_sample_array[i] is a ndarray (with dimension 12+1), corresponding to the i-th sample data point in X_sample
    # ex: X_sample_array[0] = array([ 1.       -0.993496 -0.986087 -0.807655 -0.956702 -0.948981 -0.960239 -0.986568 -0.996495 -0.718127 -1.       -0.869028 -0.137952])

    # aim: linear regression version

    # subaim: convert X_sample_array to a matrix, and y_sample to an array
    X_mat = np.array(X_sample_array)        # X_mat is a matrix with dimension N*(12+1), which means each row is a sample data point
    y_array = np.array(y_sample)            # y_array is a N*1 array, which means each element is the label of the corresponding sample data point

    # subaim: perform linear regression
    # explain: we do linear regression by using the normal equation (w = (X^TX)^(-1)X^Ty)
    # explain: we use np.linalg.inv() calculate the inverse of X^TX, and use @ to denote matrix multiplication
    w_lin = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_array

    # subaim: calculate the in-sample error
    # explain: the in-sample error is the mean squared error of the N points
    in_sample_error = np.mean((X_mat @ w_lin - y_array) ** 2)                       # ex: in_sample_error = 14.988224406335211

    # subaim: estimate the out of sample error
    # explain: the out of sample error is estimated by averaging the squared error of the rest of the 8192 - N data points
    out_of_sample_error = 0
    for i in range(len(X)):
        if i not in random_sample_indices:
            out_of_sample_error += (X_sample_array[i] @ w_lin - y[i]) ** 2
    out_of_sample_error /= (len(X) - N)
    in_out_sample_error.append((in_sample_error, out_of_sample_error))


    # aim: stochastic gradient descent (SGD) version
    for iteration in tqdm(range(1, 2)):                # note: change the number of iterations to 100000
        # subaim: pick one example uniformly at random from the "random_sample_indices"
        random_index = np.random.choice(random_sample_indices)
        x_i = X[random_index]           # x_i is a dictionary
        y_i = y[random_index]           # y_i is a float    

        # subaim: convert x_i to a ndarray "input_vector", with dimension 12+1
        # ex: input_vector = array([ 1.        -1.        -1.        -0.873062  -0.959714  -0.986787  -0.980119  -0.993284  -0.798512  -0.99425   -0.999858  -0.0106916  0.661686 ])
        input_vector = np.concatenate((np.array([1]), np.zeros(12)))
        for index, value in x_i.items():
            input_vector[index] = value

        # subaim: calculate the negative stochastic gradient, i.e. eta * sigmoid(-y_i * (w_t^T * input_vector)) * y_i * input_vector
        negative_stochastic_gradient = eta * sigmoid(-y_i * (w_t @ input_vector)) * y_i * input_vector

        w_t = w_t + negative_stochastic_gradient

        # subaim: accumulate E_in(w_t) and E_out(w_t) when t is a multiple of 200
        if iteration % 200 == 0 and iteration != 0:






# aim: for every t = 200, 400,... calculate the average (E_in(w_t), E_out(w_t)) over the 1126 experiments

# aim: calculate average (E_in(w_lin), E_out(w_lin)) over the 1126 experiments
average_in_sample_error = np.mean([error[0] for error in in_out_sample_error])
average_out_of_sample_error = np.mean([error[1] for error in in_out_sample_error])

# aim: plotting the figure
# subaim: plot the average E_in(w_t) and E_out(w_t) as a function of t
# subaim: show the average E_in(w_lin) and E_out(w_lin) as horizontal lines on the same figure