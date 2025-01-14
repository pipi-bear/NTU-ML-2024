import bz2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# aim: import file and basic setup
# decompress the file and read its content
with bz2.BZ2File("rcv1_train.binary.bz2", "rb") as f:
    file_content = f.read()

# convert bytes to string 
file_content_str = file_content.decode('utf-8')

# split the content into lines
lines = file_content_str.splitlines()

# aim: define the function that calculates the dot product of the weight vector and the input vector, and returns the sign of the dot product
def dot_product_sign(sparse_vector, w):
    total = 0
    for item in sparse_vector:
        index, value = item.split(':')
        index = int(index) 
        total += float(value) * w[index]
    if total > 0:
        return 1
    else:
        return -1

# aim: define a function that generates an infinite random sequence
def infinite_random_sequence(seed, N):

    rng = np.random.default_rng(seed)
    
    while True:
        yield rng.integers(0, N)

y_arr = []
sparse_vector_arr = []
N = 200

# aim: generate the first 200 data
for index, line in enumerate(lines):
    if index >= N:
        break
    
    parts = line.split()
    label = int(parts[0])                       # save the y value in int format in "label"
    y_arr.append(1 if label == 1 else -1)       # append the y value to y_arr
    sparse_vector = ["0:1"] + parts[1:]         # save the input vector in "sparse_vector" while eliminating the y value and adding the bias term
    sparse_vector_arr.append(sparse_vector)


update_times_arr = []           # the array to store the amount of updates until 1000 consecutive correct predictions for each experiment

# aim: repeat the experiment 1000 times
for experiment_no in tqdm(range(1)):
    w = np.zeros(47206)
    correct_amount = 0          # the amount of correct predictions, this value is reseted to 0 if there's a wrong prediction
    current_update_times = 0    # the total amount of updates until 1000 consecutive correct predictions
    seed = experiment_no

    rng = np.random.default_rng(seed)

    while correct_amount < 1000:
        random_no = rng.integers(0, N)
        sparse_vector = sparse_vector_arr[random_no]
        label = y_arr[random_no]
        mispredict = False

        sign = dot_product_sign(sparse_vector, w)
        if sign != label:
            mispredict = True

        while True:
            for item in sparse_vector:
                index, value = item.split(':')
                index = int(index) 
                w[index] += label * float(value)
            current_update_times += 1
            correct_amount = 0
            sign = dot_product_sign(sparse_vector, w)

            if sign == label:
                mispredict = False                
                correct_amount += 1
                break

    update_times_arr.append(current_update_times)

median_updates = np.median(update_times_arr)

plt.figure(figsize=(10, 6))
plt.hist(update_times_arr, bins='auto', edgecolor='black')
plt.axvline(median_updates, color='r', linestyle='dashed', linewidth=2)
plt.text(median_updates, plt.ylim()[1]*0.9, f'Median: {median_updates:.0f}', 
         horizontalalignment='center', color='r')
plt.xlabel('Number of Updates')
plt.ylabel('Frequency')
plt.title('Distribution of Update Counts')
plt.grid(True, alpha=0.3)
plt.show()

