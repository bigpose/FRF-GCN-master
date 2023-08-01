# 操作人员：徐成龙
# 操作时间：2023/5/28 15:42
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
        N, C, T, V, M = data_jpt_motion.shape   # (56880， 3， 150， 25， 2)
        data_jpt_bone_motion = np.concatenate((data_jpt_motion, data_bone_motion), axis=1)  # 将关节运动和骨骼运动的矩阵在第一维度上进行拼接
        np.save('../data/{}/{}_data_joint_bone_motion.npy'.format(dataset, set), data_jpt_bone_motion)
