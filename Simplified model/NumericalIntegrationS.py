# Numerical Integration class Alexander Batchelor

import numpy as np
import numpy.typing as npt
from math import sqrt
from scipy import sparse

def impl_euler(pos0: npt.ArrayLike, vel0: npt.ArrayLike, mkite: float, f_res: npt.ArrayLike, dt: float = 0.01):

    # dv = vel0/sqrt(vel0[0]**2 + vel0[1]**2)
    v0 = np.array([[vel0[0], 0], [0, vel0[1]]])
    dFdv = np.empty([2, 2])
    dFdx = np.empty([2, 2])
    F = np.array([[f_res[0], 0], [0, f_res[1]]])
    m = np.array([[mkite, 0], [0, mkite]])              # mass matrix
    i = np.array([[1, 0], [0, 1]])                      # identity matrix

    A = i - dt*m*dFdv - dt**2*m*dFdx
    b = dt*m*(F + dt*dFdx*v0)
    x = numpy.empty([2, 2])
    tol = sqrt(vel0[0]**2 + vel0[1]**2)/1000            # error tolerance (0.1 % of velocity)
    maxit = 100                                         # max iterations

    info, n_iter, relres = krylov.pcg(A, b, x, tol, maxit)
    return info, n_iter, relres

class NumInt:
    def __init__(self):
        pass

    def __str__(self):
        return


if __name__ == "__main__":
    pos = np.array([20, 60])
    vel = np.array([-3, 1])
    f_res = np.array([])
    mkite = 10
    dt = 0.01
    print(impl_euler(pos, vel, f_res, dt, mkite))