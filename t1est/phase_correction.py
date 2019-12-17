'''Preprocessing step for phase-sensitive reconstruction.'''

import numpy as np

def _phase_correction(x, time_axis=-1):
    '''Make complex time curves real-valued.'''

    # Move time to the back
    x = np.moveaxis(x, time_axis, -1)

    # For each time curve, find the the complex exponential that
    # makes each element of the time curve as real as possible
    R = np.median(np.angle(x), axis=-1)

    # Take the real part of the unwound time curves to have a good
    # estimate of the correct sign
    sgns = np.sign((x*np.exp(-1j*R[..., None])).real)

    # Return the real-valued signal with corrected signs
    return np.moveaxis(np.abs(x)*sgns, -1, time_axis)

if __name__ == '__main__':
    pass
