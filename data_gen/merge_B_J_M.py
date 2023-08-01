# 操作人员：徐成龙
# 操作时间：2023/6/2 9:59
# 融合骨骼和关节运动
import os
import numpy as np

sets = {
    'train', 'val'
}

datasets = {'ntu/xsub', 'ntu/xview'}

for dataset in datasets:
    for set in sets:
        print(dataset, set)
        data_jpt = np.load('../data/{}/{}_data_bone.npy'.format(dataset, set))
        data_bone = np.load('../data/{}/{}_data_joint_motion.npy'.format(dataset, set))
        N, C, T, V, M = data_jpt.shape
        data_jpt_bone = np.concatenate((data_jpt, data_bone), axis=1)
        np.save('../data/{}/{}_data_B_J_M.npy'.format(dataset, set), data_jpt_bone)
