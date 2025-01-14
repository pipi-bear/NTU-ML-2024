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

    # explain: using "np.random.default_rng" will create a unique random number generator for each experiment,
    # explain: this enables each generator to have its own state
    
    # note: this approach is different from using np.random.seed() with np.random.choice(),
    # note: which uses a shared random number generator for all experiments 

    rng = np.random.default_rng(seed)
    
    # explain: The yield statement suspends a function’s execution and sends a value back to the caller,
    # explain: and when the function resumes, it continues execution immediately after the last yield run.
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
w_lengths = []                  # use list w_lengths to store the lengths for w in each experiment

# aim: repeat the experiment 1000 times
for experiment_no in tqdm(range(1000)):
    w = np.zeros(47206)
    correct_amount = 0          # the amount of correct predictions, this value is reseted to 0 if there's a wrong prediction
    current_update_times = 0    # the total amount of updates until 1000 consecutive correct predictions
    seed = experiment_no

    w_length_history = [np.linalg.norm(w)]  # initialize w_length_history with initial w length

    # explain: when random.seed is called, it tells python to initialize the random number generator with a specific starting point,
    # explain: after later random.choice is called, it draws from this initialized sequence.

    # note: if random.seed is not called, Python uses its internal default state for randomness, which changes over time and across runs. 
    # note: This means random.choice() will generate different results each time, even if you run the same code.

    random_sequence = infinite_random_sequence(seed, N)

    # aim: for each experiment until 1000 consecutive correct predictions
    while correct_amount < 1000:

        # funct def: np.random.choice(a, size=None, replace=True, p=None)
        # explain: a: 1-D array-like or int, the range to sample from.
        # explain: size: int or tuple of ints, optional, the number of samples to draw from the distribution.
        # explain: replace: boolean, optional, determines whether the sample is with or without replacement.
        # explain: p: 1-D array-like, optional, representing the probabilities associated with each entry in a. If not given the sample assumes a uniform distribution over all entries in a.

        random_check_indices = [next(random_sequence) for _ in range(5 * N)]    
        # subaim: for each index in the random check indices    
        for random_no in random_check_indices:

            sparse_vector = sparse_vector_arr[random_no]
            label = y_arr[random_no]
            sign = dot_product_sign(sparse_vector, w)
            if sign != label:
                for item in sparse_vector:
                    index, value = item.split(':')
                    index = int(index) 
                    w[index] += label * float(value)
                correct_amount = 0
                current_update_times += 1
                w_length_history.append(np.linalg.norm(w))                      # append the length of w after each update
            else:
                correct_amount += 1
                

            if correct_amount == 1000:
                break

            # if the current update times is a multiple of 1000, this means that the current "random_check_indices" array has used out,
            # so we update the array with 1000 new random indices
            if current_update_times % 1000 == 0:                                        
                random_check_indices = [next(random_sequence) for _ in range(5 * N)]

    update_times_arr.append(current_update_times)
    w_lengths.append(w_length_history)

    seed += 1

min_T = min(update_times_arr) 

# aim: plot the length of w for each experiment
plt.figure(figsize=(10, 6))
for length in w_lengths:
    plot_length = min(min_T, len(length))
    plt.plot(range(1, plot_length + 1), length[:plot_length], alpha=0.1)
plt.xlabel('t')
plt.ylabel('Length of w')
plt.title(f'Length of w in Each Experiment (t = 1 to {min_T})')
plt.grid(True, alpha=0.3)
plt.show()

print(f"Minimum number of updates: {min_T}")