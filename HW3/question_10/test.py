import numpy as np

x = {1: -0.998916, 2: -1.0, 3: -0.549257, 4: -0.944654, 5: -0.973573, 6: -0.960239, 7: -0.986568, 8: -0.796451, 9: -0.936093, 10: -0.999504, 11: -0.855329, 12: -0.0931537}

array = np.concatenate((np.array([1]), np.zeros(12)))
print(array)
'''
for index, value in x.items():
    array[index] = value       

print(array)
'''