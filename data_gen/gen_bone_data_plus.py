import os
import numpy as np
from numpy.lib.format import open_memmap

paris = {
    # 'ntu/xview': (
    #     (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    #     (13, 1),
    #     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
    #     (25, 12)            # 关节点间的可连接方式，加了21和21自己相连接的方式
    # ),
    # 'ntu/xsub': (
    #     (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    #     (13, 1),
    #     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
    #     (25, 12)            # 关节点间的可连接方式，加了21和21自己相连接的方式
    # ),

    'kinetics': ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
                 (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15))
}

sets = {
    'train', 'val'
}

# 'ntu/xview', 'ntu/xsub',  'kinetics'
datasets = {
    'kinetics'
}
# bone
from tqdm import tqdm

for dataset in datasets:     # 人体关键节点的定义及其连接方式   'ntu/xview', 'ntu/xsub',
    for set in sets:         # 'train', 'val'
        print(dataset, set)
        data = np.load('../data/{}/{}_data_joint.npy'.format(dataset, set))            # 下载关节数据
        N, C, T, V, M = data.shape
        fp_sp = open_memmap(
            '../data/{}/{}_data_bone.npy'.format(dataset, set),
            dtype='float32',
            mode='w+',
            shape=(N, 3, T, V, M))

        fp_sp[:, :C, :, :, :] = data
        for v1, v2 in tqdm(paris[dataset]):      # paris是不同数据集的人关节点的（a,b）连接索引
            if dataset != 'kinetics':
                v1 -= 1    # 从1~25到0~24
                v2 -= 1    # 从1到0
            fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]    # 骨骼的长度和方向信息，二阶信息
