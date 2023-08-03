
import os
import numpy as np

sets = {
    'train', 'val'
}

datasets = {'ntu/xsub', 'ntu/xview'}

for dataset in datasets:
    for set in sets:
        print(dataset, set)
        data_jpt = np.load('../data/{}/{}_data_joint_cut_150.npy'.format(dataset, set))
        data_bone = np.load('../data/{}/{}_data_bone_cut_150.npy'.format(dataset, set))
        data_jpt_motion = np.load('../data/{}/{}_data_joint_motion_cut_150.npy'.format(dataset, set))
        data_bone_motion = np.load('../data/{}/{}_data_bone_motion_cut_150.npy'.format(dataset, set))
        N, C, T, V, M = data_jpt.shape
        data_4s = np.concatenate((data_jpt, data_bone, data_jpt_motion, data_bone_motion), axis=1)
        np.save('../data/{}/{}_data_4s.npy'.format(dataset, set), data_4s)
