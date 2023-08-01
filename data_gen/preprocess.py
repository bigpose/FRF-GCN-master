import sys
import random
import numpy as np
sys.path.extend(['../'])
from data_gen.rotation import *  # 随机旋转，数据增强
from tqdm import tqdm  # 进度条指令
fu = 300  # Uniform number of frames：统一帧数

def pre_normalization(data, zaxis=[0, 1], xaxis=[8, 4]):  # 预标准化
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    print('Fill null frames according to the displacement between frames')  # 根据帧之间的位移填充空帧
    for i_s, skeleton in enumerate(tqdm(s)):  # pad
        if skeleton.sum() == 0:
            print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        
                        fo = i_f + 1   # fo:Original frames：每个动作的原始帧数
                        DM = np.zeros((fo-1, V, C))  # DM:Displacement matrix:位移矩阵,生成f0-1块V行C列的全零矩阵
                        FF = np.zeros((fu, V, C))  # FF:Fill frame matrix:填充框架矩阵，生成填充之前的600个V行C列的全零矩阵
                        if fu % fo != 0:   # 如果fu不能整除fo，即fu不是fo的倍数
                            ff = fu // fo  # ff:The number of frames that need to be filled between frames：帧之间需要填充的帧数，ff为整数
                            fd = (fo * (ff + 1)) - fu  # fd:Number of frames that need to be discarded：需要丢弃的帧数
                        else:    # 如果fu是fo的倍数
                            ff = fu // fo - 1
                            fd = 0  

                        for t in range(fo):    # 用t来遍历矩阵的个数
                            for v in range(V):  # 用v来遍历行数
                                for c in range(C):   # 用c来遍历列数
                                    if t < i_f:   # 如果t在原始帧数之前
                                        DM[t, v, c] = person[t + 1, v, c] - person[t, v, c]   # 位移矩阵等于原始矩阵中相邻两帧的差
                                        Nid = DM[t, v, c] / (ff + 1)   # Nid:New inter-frame displacement：新的帧间位移,就是原先的帧间位移除以插帧之后的帧间距的数量，由原先的1变为ff+1
                                    else:  # 如果t在原始帧数之后，即最后一帧
                                        Nid = DM[t-1, v, c] / (ff + 1)  # Nid:New inter-frame displacement：新的帧间位移

                                    for f in range(ff+1):
                                        pnl = 0  # The proportion of noise level：噪声级比例
                                        Nc = f * Nid * (1+pnl) + person[t, v, c]  # Nc:New coordinates：新坐标
                                        if (fd == 0) or ((f+t*(ff+1)) <= (fu-1)): 
                                            FF[f + t*(ff+1), v, c] = Nc
                                        if (fd != 0) and ((f+t*(ff+1)) >= (fu-1)):
                                            if f < (ff+1-fd):
                                                FF[f + t*(ff+1), v, c] = Nc         

                        s[i_s, i_p, :] = FF
                        break

    print('sub the center joint #1 (spine joint in ntu and neck joint in kinetics)')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        main_body_center = skeleton[0][:, 1:2, :].copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    print('parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, zaxis[0]]
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

    print(
        'parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)

    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data


if __name__ == '__main__':
    data = np.load('../data/ntu/xview/val_data.npy')
    pre_normalization(data)
    np.save('../data/ntu/xview/data_val_pre.npy', data)
