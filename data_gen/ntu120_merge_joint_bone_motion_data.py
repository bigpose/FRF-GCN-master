
import os
import numpy as np

sets = {
    'train', 'val'
}

datasets = {'ntu120/xsetup', 'ntu120/xsub'}

for dataset in datasets:
    for set in sets:
        print(dataset, set)
        data_jpt_motion = np.load('../data/{}/{}_data_joint_motion.npy'.format(dataset, set))
        data_bone_motion = np.load('../data/{}/{}_data_bone_motion.npy'.format(dataset, set))
        N, C, T, V, M = data_jpt_motion.shape  
        data_jpt_bone_motion = np.concatenate((data_jpt_motion, data_bone_motion), axis=1) 
        np.save('../data/{}/{}_data_joint_bone_motion.npy'.format(dataset, set), data_jpt_bone_motion)
