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

X, y = read_libsvm_format('cpusmall_scale')


np.random.seed(42)
random_sample_indices = np.random.choice(len(X), 25, replace=False)

X_sample = [X[i] for i in random_sample_indices]
y_sample = [y[i] for i in random_sample_indices]
print(X_sample[0:3])

X_sample_array = []
for sample in X_sample:
    input_vector = np.array([1, 0, 0], dtype=float)
    for index, value in sample.items():
        if index <= 2:
            input_vector[index] = value    
    X_sample_array.append(input_vector)


print(X_sample_array[0:3])