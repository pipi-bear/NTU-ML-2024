import bz2
import numpy as np
import matplotlib.pyplot as plt

with bz2.BZ2File("rcv1_train.binary.bz2", "rb") as f:
    file_content = f.read()

file_content_str = file_content.decode('utf-8')

lines = file_content_str.splitlines()

def dot_product_sign(sparse_vector, w):
    total = 0
    for item in sparse_vector:
        index, value = item.split(':')
        index = int(index) - 1              # Adjust if indices start from 1
        total += float(value) * w[index]
        if total > 0:
            return 1
        else:
            return -1
        
def infinite_random_sequence(seed, N):

    rng = np.random.default_rng(seed)
    
    while True:
        yield rng.integers(0, N)

y_arr = []
sparse_vector_arr = []
N = 200

for index, line in enumerate(lines):
    if index >= N:
        break
    
    parts = line.split()
    label = int(parts[0])                       # save the y value in int format in "label"
    y_arr.append(1 if label == 1 else -1)       # append the y value to y_arr
    sparse_vector = ["1:1"] + parts[1:]         # save the input vector in "sparse_vector" while eliminating the y value and adding the bias term
    sparse_vector_arr.append(sparse_vector)

w = np.zeros(47206)
sparse_vector = sparse_vector_arr[0]
sign = dot_product_sign(sparse_vector, w)

if sign != y_arr[0]:
    for item in sparse_vector:
        index, value = item.split(':')
        index = int(index) - 1
        w[index] += label * float(value)

notzero_element = []
for element in w:
    if element != 0:
        notzero_element.append(element)

print(notzero_element[0:10])


