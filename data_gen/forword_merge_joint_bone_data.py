import os
import numpy as np
import torch.nn as nn
import sys
sys.path.append("..")
from model.agcn_stc_sl import *
from graph import ntu_rgb_d_A


class forword_merge(nn.Module):
    def __init__(self, in_channels=3, num_point=25, num_person=2, graph=None, graph_args=dict(), residual=True):
        super(forword_merge, self).__init__()

        graph = ntu_rgb_d_A.Graph

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        A = self.graph.A

        self.bn = nn.BatchNorm2d(num_person * in_channels * num_point)
        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 32, A)

        self.stc1 = STC_Att(64, 4)
        self.stc2 = STC_Att(32, 4)

    def forword(self, x):
        N, C, T, V, M = x.size()  # 其中N表示样本数, C表示通道数, T表示总帧数,V表示节点数,M表示视频中的人数

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)  # permute函数可以对任意高维矩阵进行转置
        x = self.bn(x)  # contiguous一般与transpose，permute,view搭配使用
        # 即使用transpose或permute进行维度变换后，调用contiguous，然后方可使用view对维度进行变形。
        x = x.view(N, M, V, C, T).permute(0, 3, 4, 2, 1)  # N,C,T,V,M
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.stc1(x)
        x = self.l3(x)
        x = self.stc2(x)

        return (x)


sets = {'train', 'val'}
datasets = {'ntu/xsub', 'ntu/xview'}

for dataset in datasets:
    for set in sets:
        print(dataset, set)
        data_jpt = np.load('../data/{}/{}_data_joint.npy'.format(dataset, set))
        data_jpt = forword_merge(data_jpt)
        data_bone = np.load('../data/{}/{}_data_bone.npy'.format(dataset, set))
        data_bone = forword_merge(data_bone)
        N, C, T, V, M = data_jpt.shape
        data_jpt_bone = np.concatenate((data_jpt, data_bone), axis=1)  # 将关节和骨骼的矩阵在第一维度上进行拼接
        np.save('../data/{}/{}_data_joint_bone.npy'.format(dataset, set), data_jpt_bone)
