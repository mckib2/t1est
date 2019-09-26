'''Show basic usage of t1est.'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from phantominator import shepp_logan
from t1est import t1est

if __name__ == '__main__':

    # Make a simple Shepp-Logan MR phantom
    N = 64
    M0 = shepp_logan(N)
    T1 = 1 + M0*2

    # Naive inversion recovery simulation for this demo
    nt = 11
    TIs = np.linspace(.01, 3*np.max(T1.flatten()), nt)
    ph = M0[..., None]*(
        1 - 2*np.exp(-TIs[None, None, :]/T1[..., None]))
    mask = np.abs(ph[..., -1]) > 1e-8

    # Get a T1 map!
    T1map = t1est(
        ph, TIs, time_axis=-1, mask=mask, method='lm',
        T1_bnds=(np.min(T1.flatten()), np.max(T1.flatten())),
        chunksize=10, molli=False, mag=False)

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
    im = plt.imshow(T1map, **plt_opts)
    plt.title('T1 estimates')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


    ax = plt.subplot(nx, ny, 3)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    im = plt.imshow(np.abs((T1map - T1)*mask))
    plt.title('Residual')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
