import numpy as np

def sample_category(batch_size, dim):
    return np.random.multinomial(1, dim * [1.0/dim], size=batch_size)


v = sample_category(5, 10)
print(v)