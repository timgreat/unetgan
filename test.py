import numpy as np

dataset = np.load('celeba128_inception_moments.npz', allow_pickle=True)
print(dataset.files)
print(len(dataset.files))