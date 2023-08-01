import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))                        # 论文中使用的N*N表示Ak，即代码中的V
    for i, j in link:                                         # link引入的是self_link，代表求出单位矩阵I
        A[j, i] = 1
    return A


def normalize_digraph(A):  # 除以每列的和（归一化）
    Dl = np.sum(A, 0)      # axis为0是压缩行,即将每一列的元素相加,将矩阵压缩为一行，Dl是w×1的向量
    h, w = A.shape         # 即代码中V*V
    Dn = np.zeros((w, w))  # 返回一个w行w列的以0填充的矩阵，w是A的列数，用来存放归一化后的对角度矩阵
    for i in range(w):
        if Dl[i] > 0:      # 如果A的列元素的和大于0，那么将其变为倒数，即归一化，Dn就是对角线元素为归一化元素的对角矩阵
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)     # 将A和Dn两个矩阵点乘为AD矩阵，这就是用Dn将A归一化后的矩阵
    return AD              # AD矩阵每一列的和为1，元素的数值即为权重，决定两个顶点之间是否存在连接，它代表人体的物理结构


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)  # 得到单位矩阵
    In = normalize_digraph(edge2mat(inward, num_node))  # 括号里的是不同关节点的关系矩阵，也就是邻接矩阵A，In是归一化后的前向邻接矩阵
    Out = normalize_digraph(edge2mat(outward, num_node))   # Out是归一化后的后向邻接矩阵
    A = np.stack((I, In, Out))
    return A      # A是I+A


def get_spatial_graph2(num_node, self_link):
    I = edge2mat(self_link, num_node)  # 得到单位矩阵
    A = I
    return A      # A是I
