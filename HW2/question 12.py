import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

E_in_all_experiments = []
E_out_all_experiments = []
difference_list = []

# aim: repeat the experiment 2000 times
for experiment_no in tqdm(range(2000)):
    x_arr = np.random.uniform(-1, 1, 12) # generate 12 x values, that are uniformly distributed in [-1, 1]
    y_arr = []
    for x_val in x_arr:
        if x_val > 0:
            y_arr.append(1)
        else:                            # assuming that sign(0) = -1
            y_arr.append(-1)

    # aim: add noise that flips the sign with 15% probability
    # explain: we generate noise that is -2y(15%) and 0(85%, which means without noise), so that when we add the noise to y,
    # explain: if y = 1, then y + noise = 1 + (-2) = -1
    # explain: if y = -1, then y + noise = (-1) + 2 = 1

    noise_arr = []
    np.random.seed(experiment_no)
    for y in y_arr:
        noise = np.random.choice([-2 * y, 0], p = [0.15, 0.85])
        noise_arr.append(int(noise))

    y_with_noise_arr = []
    for y, n in zip(y_arr, noise_arr):
        y_w_noise = y + n
        y_with_noise_arr.append(y_w_noise)

    data_points_list = list(zip(x_arr, y_with_noise_arr))
    sorted_data_points_list = sorted(data_points_list, key=lambda point: point[0])

    s = np.random.choice([-1, 1])
    theta = np.random.uniform(-1, 1)

    total_error = 0
    for x, y in sorted_data_points_list:
        if x - theta > 0:
            sign = 1
        else:
            sign = -1
        prediction = s * sign
        if prediction != y:
            total_error += 1
    avg_total_error = total_error / 12

    # aim: compute E_out(g)
    v = s * 0.35
    u = 0.5 - v
    E_out = u + v * abs(theta)

    # aim: store the result of in sample error and out of sample error, and their differences
    E_in_all_experiments.append(avg_total_error)
    E_out_all_experiments.append(E_out)
    difference_list.append(E_out - avg_total_error)

median_difference = np.median(difference_list)


plt.figure(figsize=(10, 6))
plt.scatter(E_in_all_experiments, E_out_all_experiments, alpha=0.5)
plt.xlabel('E_in')
plt.ylabel('E_out')
plt.title('Scatter plot of E_in and corresponding E_out for 2000 experiments')
plt.grid(True)

plt.text(0.05, 0.95, f'Median(E_out - E_in) = {median_difference:.6f}', 
         transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()
plt.show()

print(f"Median of E_out - E_in: {median_difference:.6f}")

