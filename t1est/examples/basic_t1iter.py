'''Example usage of iterative estimator.'''

from time import time
# from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from phantominator import shepp_logan

from t1est import t1iter

# def _show_result(T1, A, B, T1_est, A_est, B_est, niter, time_elapsed):
#     print('Took %g sec and %d iterations' % (time_elapsed, niter))
#     print('    True T1: %g' % T1)
#     print('     est T1: %g' % T1_est)
#     print('     True A: %g' % A)
#     print('      est A: %g' % A_est)
#     print('     True B: %g' % B)
#     print('      est B: %g' % B_est)
#
#     plt.plot(err)
#     plt.title('dT1 vs iteration')
#     plt.xlabel('Iteration')
#     plt.ylabel('Change in T1 estimate')
#     plt.show()

if __name__ == '__main__':

    # Make a simple Shepp-Logan MR phantom
    N = 64
    M0 = shepp_logan(N)
    T1 = 1 + M0*2

    # Naive inversion recovery simulation for this demo
    nt = 11
    TIs = np.linspace(.01, 5*np.max(T1.flatten()), nt)
    ph = M0[..., None]*(
        1 - 2*np.exp(-TIs[None, None, :]/T1[..., None]))
    mask = np.abs(ph[..., -1]) > 1e-8

    # Do the estimate, can tune alpha, tol to get quicker, better
    # estimates:
    t0 = time()
    T1_est, A_est, B_est, niter, err = t1iter(
        ph, TIs, T10=1, A0=1, B0=-2, time_axis=-1, alpha=1.75,
        tol=1e-5, maxiter=1000)
    print('Took %d iters in %g sec' % (niter, time() - t0))

    # Look at error
    plt.semilogy(err)
    plt.xlabel('Iteration')
    plt.ylabel('log(max(abs(dT1)))')
    plt.title('Change in T1 estimate vs iteration')
    plt.show()

    # Check it out
    nx, ny = 1, 3
    cbar_opts = {'size':"5%", 'pad':0.05}
    plt_opts = {'vmin': 0, 'vmax': 3}

    ax = plt.subplot(nx, ny, 1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    im = plt.imshow(T1*mask, **plt_opts)
    plt.title('Target T1 map')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', **cbar_opts)
    plt.colorbar(im, cax=cax)

    ax = plt.subplot(nx, ny, 2)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    im = plt.imshow(T1_est*mask, **plt_opts)
    plt.title('T1 estimates')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax = plt.subplot(nx, ny, 3)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    im = plt.imshow(np.abs((T1_est - T1)*mask))
    plt.title('Residual')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
