import numpy as np
A = [1, 2, 2, 5, 3, 4, 3]

a, s= np.unique(A, return_index=True)
print(a)
print(s)
print('###########')

a, s, p = np.unique(A, return_inverse=True)
print(a)
print(s)
print(p)
#
