import random
import torch
import numpy as np


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M Randomly selecting one of the paragraphs doesn't make a lot of sense. Because there are 0
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])  # xuanzhuan juzhen

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]  # pingyi bianhuan
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    # input: C,T,V,M Offset one of the segments
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift

def _rot1(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros),dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim = 1)  # T,3,3

    ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 1)

    rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 1)

    rot = rz.matmul(ry).matmul(rx)
    return rot

def _rot2(rot):
    """
    rot: T,6
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,6
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros, zeros, zeros, zeros), dim=-1)  # T,1,6
    rx2 = torch.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1], zeros, zeros, zeros), dim=-1)  # T,1,6
    rx3 = torch.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1], zeros, zeros, zeros), dim=-1)  # T,1,6
    rx = torch.cat((r1, rx2, rx3, zeros, zeros, zeros), dim=1)  # T,3,3

    ry1 = torch.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2], zeros, zeros, zeros), dim=-1)
    r2 = torch.stack((zeros, ones, zeros, zeros, zeros, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2], zeros, zeros, zeros), dim=-1)
    ry = torch.cat((ry1, r2, ry3, zeros, zeros, zeros), dim=1)

    rz1 = torch.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros, zeros, zeros, zeros), dim=-1)
    r3 = torch.stack((zeros, zeros, ones, zeros, zeros, zeros), dim=-1)
    rz2 = torch.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros, zeros, zeros, zeros), dim=-1)
    rz = torch.cat((rz1, rz2, r3, zeros, zeros, zeros), dim=1)

    rot = rz.matmul(ry).matmul(rx)
    return rot

def _rot(rot):
    """
    rot: T,6
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,6
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros, zeros, zeros, zeros), dim=-1)  # T,1,6
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1], zeros, zeros, zeros), dim=-1)  # T,1,6
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1], zeros, zeros, zeros), dim=-1)  # T,1,6
    rx4 = torch.stack((zeros, zeros, zeros, ones, zeros, zeros), dim=-1)  # T,1,6
    rx5 = torch.stack((zeros, zeros, zeros, zeros, cos_r[:,1:2], sin_r[:,1:2]), dim=-1)  # T,1,6
    rx6 = torch.stack((zeros, zeros, zeros, zeros, -sin_r[:,1:2], cos_r[:,1:2]), dim=-1)  # T,1,6
    rx = torch.cat((r1, rx2, rx3, rx4, rx5, rx6), dim=1)  # T,6,6

    ry1 = torch.stack((cos_r[:,2:3], zeros, -sin_r[:,2:3], zeros, zeros, zeros), dim=-1)
    ry2 = torch.stack((zeros, cos_r[:,3:4], sin_r[:,3:4], zeros, zeros, zeros), dim=-1)
    ry3 = torch.stack((sin_r[:,2:3], zeros, cos_r[:,2:3], zeros, zeros, zeros), dim=-1)
    ry4 = torch.stack((zeros, zeros, zeros, cos_r[:,3:4], zeros, -sin_r[:,3:4]), dim=-1)
    r5 = torch.stack((zeros, ones, zeros, zeros, zeros, zeros), dim=-1)
    ry6 = torch.stack((zeros, zeros, zeros, zeros, sin_r[:,3:4], cos_r[:,3:4]), dim=-1)
    ry = torch.cat((ry1, ry2, ry3, ry4, r5, ry6), dim=1)

    rz1 = torch.stack((cos_r[:,4:5], sin_r[:,4:5], zeros, zeros, zeros, zeros), dim=-1)
    rz2 = torch.stack((-sin_r[:,4:5], cos_r[:,4:5], zeros, zeros, zeros, zeros), dim=-1)
    rz3 = torch.stack((zeros, zeros, ones, zeros, zeros, zeros), dim=-1)
    rz4 = torch.stack((zeros, zeros, zeros, cos_r[:,5:6], sin_r[:,5:6], zeros), dim=-1)
    rz5 = torch.stack((zeros, zeros, zeros, -sin_r[:, 5:6], cos_r[:, 5:6], zeros), dim=-1)
    rz6 = torch.stack((zeros, zeros, zeros, zeros, zeros, ones), dim=-1)
    rz = torch.cat((rz1, rz2, rz3, rz4, rz5, rz6), dim=1)

    rot = rz.matmul(ry).matmul(rx)
    return rot

def random_rot(data_numpy, theta=0.1):
    """
    data_numpy: C,T,V,M
    """
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V*M)  # T,3,V*M
    rot = torch.zeros(6).uniform_(-theta, theta)
    rot = torch.stack([rot, ] * T, dim=0)
    rot = _rot(rot)  # T,6,6
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

    return data_torch

def random_rot1(data_numpy, theta=0.3):
    """
    data_numpy: C,T,V,M
    """
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V*M)  # T,3,V*M
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot, ] * T, dim=0)
    rot = _rot(rot)  # T,3,3
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

    return data_torch

def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy
