import math
import numpy as np
import cupy as cp

# --- Lattice parameters---
mu2 = 1.0
l = 32
N = 512
a = l / N
a2 = a**2
Ny = 50
k_IR = 2 * math.pi / l
m2 = (0.5 * k_IR)**2
Nc = 3
alpha_fc = 1.0
dY = 0.01
Y_f = 1.0

variance_of_mv_noise = math.sqrt(mu2 / (Ny * a2))


# --- Gell-Mann generators (t^a = lambda^a / 2, plus t^9 for completeness) ---

def _build_generators():
    t = np.zeros((9, Nc, Nc), dtype=np.complex64)

    t[0, 0, 1] = 0.5
    t[0, 1, 0] = 0.5

    t[1, 0, 1] = -0.5j
    t[1, 1, 0] = 0.5j

    t[2, 0, 0] = 0.5
    t[2, 1, 1] = -0.5

    t[3, 0, 2] = 0.5
    t[3, 2, 0] = 0.5

    t[4, 0, 2] = -0.5j
    t[4, 2, 0] = 0.5j

    t[5, 1, 2] = 0.5
    t[5, 2, 1] = 0.5

    t[6, 1, 2] = -0.5j
    t[6, 2, 1] = 0.5j

    t[7, 0, 0] = 1.0 / (2.0 * np.sqrt(3.0))
    t[7, 1, 1] = 1.0 / (2.0 * np.sqrt(3.0))
    t[7, 2, 2] = -1.0 / np.sqrt(3.0)

    t[8] = np.eye(Nc, dtype=np.complex64) * np.sqrt(2.0 / 3.0) / 2.0

    return cp.asarray(t)


generators = _build_generators()


def validate_orthonormality(t=None):
    if t is None:
        t = generators
    t_np = t.get()
    max_err = 0.0
    for a in range(9):
        for b in range(9):
            val = np.trace(t_np[a] @ t_np[b])
            expected = 0.5 if a == b else 0.0
            err = abs(val - expected)
            if err > max_err:
                max_err = err
    return max_err
