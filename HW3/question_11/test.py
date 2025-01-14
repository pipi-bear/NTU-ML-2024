import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

N_list = list(range(25, 2025, 25))
seed = 0

for N in tqdm(N_list):
    for experiment in range(16):
        seed += 1
        np.random.seed(seed)
        random_sample_indices = np.random.choice(8192, N, replace=False)

        if seed % 16 == 0:
            print(random_sample_indices[:5])

print(seed)