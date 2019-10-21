'''Helper func.'''

import numpy as np
from scipy.io import loadmat

def loader(idx):
    '''Load data, inversion times, and truth from MAT files.'''

    basename = 'data/Pre_contrast_MOLLI_T1_testing'
    filenames = [
        'P072919_T1Map_LongT1_2CH_MOCO_19.mat',
        'P072919_T1Map_LongT1_4CH_MOCO_16.mat',
        'P072919_T1Map_LongT1_SA_Apex_MOCO_13.mat',
        'P072919_T1Map_LongT1_SA_Base_MOCO_7.mat',
        'P072919_T1Map_LongT1_SA_Mid_MOCO_10.mat',
        'P082119_T1Map_LongT1_4CH_MOCO_16.mat',
        'P082119_T1Map_LongT1_SA_Apex_MOCO_13.mat',
        'P082119_T1Map_LongT1_SA_Apex_MOCO_45.mat',
        'P082119_T1Map_LongT1_SA_Base_MOCO_7.mat',
        'P082119_T1Map_LongT1_SA_Mid_MOCO_10.mat',
        'P082119_T1Map_LongT1_SA_Mid_MOCO_42.mat',
        'P082819_T1Map_LongT1_4CH_MOCO_20.mat',
        'P082819_T1Map_LongT1_SA_Apex_MOCO_17.mat',
        'P082819_T1Map_LongT1_SA_Base_MOCO_8.mat',
        'P082819_T1Map_LongT1_SA_Mid_MOCO_11.mat',
        'P082819_T1Map_LongT1_SA_Mid_MOCO_14.mat',
    ]

    data = loadmat(basename + '/' + filenames[idx])
    TIs = data['inv_times'].squeeze()

    # remove all other keys except the one we want
    for key in [
            '__version__', '__header__', '__globals__', 'inv_times']:
        del data[key]

    # The only remaining key should be the data
    key = list(data.keys())[0]

    # First eight are the images corresponding to inversion times
    ims = data[key][..., :8].astype(np.double)/1000

    # Last is the reference T1
    truth = data[key][..., -1].astype(np.double)/1000

    # Sort TIs and ims to be chronological
    ind = np.argsort(TIs)
    TIs = np.sort(TIs)
    ims = ims[..., ind]

    # We want TIs in seconds
    TIs = TIs*1e-3

    return(TIs, ims, truth, filenames[idx][:-4])
