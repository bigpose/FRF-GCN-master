# 操作人员：徐成龙
# 操作时间：2023/5/28 15:31
import os
import numpy as np
from numpy.lib.format import open_memmap

paris = {
    'ntu120/xsetup': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
    'ntu120/xsub': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
}

sets = {'train', 'val'}

# 'ntu/xview', 'ntu/xsub'
datasets = {'ntu120/xsetup', 'ntu120/xsub'}
# 骨骼生成
from tqdm import tqdm

for dataset in datasets:     # 人体关键节点的定义及其连接方式   'ntu/xsub','ntu/xview'
    for set in sets:         # 'train', 'val'
        print(dataset, set)
        data = np.load('../data/{}/{}_data_joint.npy'.format(dataset, set))            # 加载关节数据到data中
        N, C, T, V, M = data.shape
        fp_sp = open_memmap(
            '../data/{}/{}_data_bone.npy'.format(dataset, set),
            dtype='float32',
            mode='w+',
            shape=(N, 3, T, V, M))

        fp_sp[:, :C, :, :, :] = data
        for v1, v2 in tqdm(paris[dataset]):      # paris是不同数据集的人关节点的（a,b）连接索引，v1和v2代表的是两个连接的关节的代号
            if dataset != 'kinetics':
                v1 -= 1    # 从1~25到0~24
                v2 -= 1    # 从1到0
            fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]    # 骨骼的长度和方向信息，二阶信息
