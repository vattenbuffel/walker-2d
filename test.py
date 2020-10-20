import numpy as np 


def clip(r,A):
    return np.minimum(r*A, np.clip(r, 0.8, 1.2)*A)

def clip_noa(r,a):
    return np.clip(r, 0.8, 1.2)*A

r = 10
A = -1

print(clip(r,A))
print(clip_noa(r,A))