import sys

sys.path.extend(['../'])
from graph import tools

num_node = 25         # 关节点数为25（NTU-RGB+D）
self_link = [(i, i) for i in range(num_node)]    # self_link代表相同关节点的连接，其实就是为了构建单位矩阵I
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]     # 关节点间的可连接方式，没有21和21自己相连的方式
                                                                         # 其实就是为了构建不同关节之间的连接
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]                 # 为了从0开始
outward = [(j, i) for (i, j) in inward]                                  # 反过来，为了构建无向图
neighbor = inward + outward                                              # 正向和反向加起来


class Graph:                             # 就是为了得到邻接矩阵A
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph2(num_node, self_link)  # A就是原论文中的Ak,代表人体的物理连接结构
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)
