import sys
import random
import numpy as np
sys.path.extend(['../'])
from data_gen.rotation import *
from tqdm import tqdm  
fu = 300  

def pre_normalization(data, zaxis=[0, 1], xaxis=[8, 4]):  
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    print('Fill null frames according to the displacement between frames')  
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
                        
                        fo = i_f + 1   # fo:Original frames：Raw frames per action
                        DM = np.zeros((fo-1, V, C))  # DM:Displacement matrix:Displacement matrix, generating all-zero matrices with V rows and C columns for the f0-1 block
                        FF = np.zeros((fu, V, C))  # FF:Fill frame matrix:Fill the frame matrix to generate an all-zero matrix of 600 V rows and C columns prior to the fill
                        if fu % fo != 0:   # If fu is not divisible by fo, i.e., fu is not a multiple of fo
                            ff = fu // fo  # ff:The number of frames that need to be filled between frames：Number of frames to fill between frames, ff is an integer
                            fd = (fo * (ff + 1)) - fu  # fd:Number of frames that need to be discarded：Number of frames to be discarded
                        else:    # If fu is a multiple of fo
                            ff = fu // fo - 1
                            fd = 0  

                        for t in range(fo):    # Iterate over the number of matrices with t
                            for v in range(V):  # Iterate through the rows with v
                                for c in range(C):   # Iterate over columns with c
                                    if t < i_f:   # If t is before the original frame number
                                        DM[t, v, c] = person[t + 1, v, c] - person[t, v, c]   # The displacement matrix is equal to the difference between two neighboring frames in the original matrix
                                        Nid = DM[t, v, c] / (ff + 1)   # Nid:New inter-frame displacement：The new inter-frame displacement, which is the original inter-frame displacement divided by the number of frame spacing after frame insertion, is changed from 1 to ff+1.
                                    else:  # If t is after the original number of frames, i.e., the last frame
                                        Nid = DM[t-1, v, c] / (ff + 1)  # Nid:New inter-frame displacement：New inter-frame displacement

                                    for f in range(ff+1):
                                        pnl = 0  # The proportion of noise level：Proportion of noise level
                                        Nc = f * Nid * (1+pnl) + person[t, v, c]  # Nc:New coordinates：new coordinate
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
