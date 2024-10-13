import numpy as np
import scipy

def gaussian_transition(t):
    r"""Computes transition matrix for q(x_t|x_{t-1}).

    This method constructs a transition matrix Q with
    decaying entries as a function of how far off diagonal the entry is.
    Normalization option 1:
    Q_{ij} =  ~ softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                1 - \sum_{l \neq i} Q_{il}  if i==j.
                0                          else.

    Normalization option 2:
    tilde{Q}_{ij} =  softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                        0                        else.

    Q_{ij} =  tilde{Q}_{ij} / sum_l{tilde{Q}_{lj}}

    Args:
        t: timestep. integer scalar (or numpy array?)

    Returns:
        Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    betas = np.linspace(0.02, 1, 1000)

    num_pixel_vals = 32

    transition_bands = num_pixel_vals - 1

    beta_t = betas[t]

    mat = np.zeros((num_pixel_vals, num_pixel_vals),
                    dtype=np.float64)

    # Make the values correspond to a similar type of gaussian as in the
    # gaussian diffusion case for continuous state spaces.
    values = np.linspace(start=0., stop=255., num=num_pixel_vals,
                            endpoint=True, dtype=np.float64)
    values = values * 2./ (num_pixel_vals - 1.)
    values = values[:transition_bands+1]
    values = -values * values / beta_t

    values = np.concatenate([values[:0:-1], values], axis=0)
    values = scipy.special.softmax(values, axis=0)
    values = values[transition_bands:]
    for k in range(1, transition_bands + 1):
        off_diag = np.full(shape=(num_pixel_vals - k,),
                            fill_value=values[k],
                            dtype=np.float64)

        mat += np.diag(off_diag, k=k)
        mat += np.diag(off_diag, k=-k)

    # Add diagonal values such that rows and columns sum to one.
    # Technically only the ROWS need to sum to one
    # NOTE: this normalization leads to a doubly stochastic matrix,
    # which is necessary if we want to have a uniform stationary distribution.
    diag = 1. - mat.sum(1)
    mat += np.diag(diag, k=0)

    return mat


if __name__ == "__main__":
    gaussian_transition_matrix = gaussian_transition(888)
    import matplotlib.pyplot as plt

    # 可视化高斯过渡矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(gaussian_transition_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Gaussian Transition Matrix')
    plt.savefig('gaussian_transition_matrix.png')