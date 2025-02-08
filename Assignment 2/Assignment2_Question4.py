import matplotlib

import numpy as np


A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])

A_inverse = np.linalg.inv(A)
id1 = np.dot(A, A_inverse)
id2 = np.dot(A_inverse, A)

print("Inverse of A:")
print(A_inverse)

print("\nA * A^-1:")
print(id1)

print("\nA^-1 * A:")
print(id2)