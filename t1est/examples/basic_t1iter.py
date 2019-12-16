'''Example usage of iterative estimator.'''

from time import time
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from t1est import t1iter

def _show_result(T1, A, B, T1_est, A_est, B_est, niter, time_elapsed):
    print('Took %g sec and %d iterations' % (time_elapsed, niter))
    print('    True T1: %g' % T1)
    print('     est T1: %g' % T1_est)
    print('     True A: %g' % A)
    print('      est A: %g' % A_est)
    print('     True B: %g' % B)
    print('      est B: %g' % B_est)

    plt.plot(err)
    plt.title('dT1 vs iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Change in T1 estimate')
    plt.show()

if __name__ == '__main__':

    # Generate a simple inversion recovery signal
    T1 = 1.8 # in sec
    t = np.linspace(.1, 6, 8) # in sec
    A, B = 1, -2
    s = A + B*np.exp(-t/T1)
    _show_resultp = partial(_show_result, T1=T1, A=A, B=B)

    # Best usage is if we have a good idea what parameters are
    t0 = time()
    T1_est, A_est, B_est, niter, err = t1iter(
        s, t, T10=1, A0=1, B0=-2)
    time_elapsed = time() - t0
    _show_resultp(
        T1_est=T1_est, A_est=A_est, B_est=B_est, niter=niter,
        time_elapsed=time_elapsed)

    # We might accidentally start with bad estimates, in this case
    # we'll need a few more iterations
    t0 = time()
    T1_est, A, B, niter, err = t1iter(
        s, t, T10=10, A0=-2, B0=0, maxiter=1000)
    time_elapsed = time() - t0
    _show_resultp(
        T1_est=T1_est, A_est=A_est, B_est=B_est, niter=niter,
        time_elapsed=time_elapsed)
