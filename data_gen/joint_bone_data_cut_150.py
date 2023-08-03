
import os
import numpy as np
from numpy.lib.format import open_memmap

sets = {
    'train', 'val'
}

# 'ntu/xview', 'ntu/xsub'
datasets = {
    'ntu/xview', 'ntu/xsub',
}

from tqdm import tqdm

for dataset in datasets:
    for set1 in sets:
        print(dataset, set1)
        data = np.load('../data/{}/{}_data_joint_bone.npy'.format(dataset, set1))
        N, C, T, V, M = data.shape
        T1 = T // 2
        reverse = open_memmap(
            '../data/{}/{}_data_joint_bone_cut_150.npy'.format(dataset, set1),
            dtype='float32',
            mode='w+',
            shape=(N, 6, T1, V, M))
        reverse[:, :, :T1, :, :] = data[:, :, ::2, :, :]
