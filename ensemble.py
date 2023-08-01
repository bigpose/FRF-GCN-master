import argparse
import pickle

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='ntu/xsub', choices={'ntu/xsub', 'ntu/xview'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=1, help='weighted summation')
arg = parser.parse_args()

dataset = arg.datasets
label = open('./data/' + dataset + '/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open('./work_dir/' + dataset + '/agcn_test_joint_bone/epoch1_test_score.pkl', 'rb')  # r1是关节和骨骼融合流的测试分数
r1 = list(pickle.load(r1).items())
r2 = open('./work_dir/' + dataset + '/agcn_test_joint_bone_motion/epoch1_test_score.pkl', 'rb')  # r2是关节运动和骨骼运动融合流的测试分数
r2 = list(pickle.load(r2).items())
right_num = total_num = right_num_5 = 0  # 初始化三个数为0
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    r = r11 + r22 * arg.alpha
    rank_5 = r.argsort()[-5:]
    right_num_5 += int(int(l) in rank_5)
    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num  # 计算top-1准确率
acc5 = right_num_5 / total_num  # 计算top-5准确度
print(acc, acc5)