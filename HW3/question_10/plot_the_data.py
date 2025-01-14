import matplotlib.pyplot as plt
import numpy as np

def read_libsvm_format(file_path):
    X, y = [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            y.append(float(parts[0]))  # Label is the first element
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

# aim: Loop through each feature and plot
num_features = len(X[0])

fig, axs = plt.subplots(4, 3, figsize=(12, 12))                      # Create a 4x3 grid of subplots

for feature in range(1, num_features + 1):
    feature_values = [x.get(feature, None) for x in X]               # Extract the values of the current feature
    
    # Calculate the row and column index for the current subplot
    row = (feature - 1) // 3
    col = (feature - 1) % 3
    
    # Create a scatter plot for the current feature in the appropriate subplot
    axs[row, col].scatter(feature_values, y, s=5, alpha = 0.7)
    axs[row, col].set_xlabel(f'Feature {feature}', fontsize=8)
    axs[row, col].set_ylabel('y', fontsize=8)
    axs[row, col].grid(True, alpha = 0.3)
    axs[row, col].tick_params(axis='both', which='major', labelsize=8)  

plt.tight_layout(pad = 1)
plt.show()