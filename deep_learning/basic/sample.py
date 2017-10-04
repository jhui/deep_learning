import numpy as np

def sample_category(batch_size, dim):
    indices = np.random.randint(0, dim, size=batch_size)

    as_one_hot = np.zeros((indices.shape[0], dim))
    as_one_hot[np.arange(0, indices.shape[0]), indices] = 1.0

    return as_one_hot


v = sample_category(5, 10)
print(v)