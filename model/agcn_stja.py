# 操作人员：徐成龙
# 操作时间：2023/3/21 15:24
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):           # unit_tcn和unit_gcn对应layer部分
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))  # kernel_size为9*1的卷积核

        self.bn = nn.BatchNorm2d(out_channels)    # 批量归一化
        self.relu = nn.ReLU()                     # 激活函数Relu
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):      # spatial GCN + bn + relu
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding     # python中“//”是一个算术运算符，表示整数除法，它可以返回商的整数部分（向下取整）
        self.inter_c = inter_channels
        # self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        # 将一个不可训练的类型Tensor转换成可以训练的类型parameter，成为了模型中根据训练可以改动的参数
        # nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        # A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)  # 论文中的N*CT
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)          # 论文中的CT*N
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V  # torch.matmul，实现张量矩阵A1和A2的乘积
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

# 联合TCN和GCN
class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)

class ST_Joint_Att(nn.Module):
    def __init__(self, channel, reduct_ratio, bias=True, **kwargs):
        super(ST_Joint_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),)
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)                  # 输入特征在时间维度被平均
        x_v = x.mean(2, keepdims=True).transpose(2, 3)  # 输入特征在空间维度被平均
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))  # 将平均后的信息拼接在一起，通过fc层馈送压缩信息
        x_t, x_v = torch.split(x_att, [T, V], dim=2)    # 再将其分成时间和空间
        x_t_att = self.conv_t(x_t).sigmoid()            # 恢复通道维数，再由sigmoid激活
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att                       # 求两者通道外积，由原先的特征图得到注意力图x_att
        return x_att


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=6):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.stja1 = ST_Joint_Att(128, 4)
        self.stja2 = ST_Joint_Att(128, 4)
        self.stja3 = ST_Joint_Att(128, 4)

        self.l1 = TCN_GCN_unit(6, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()    # 其中N表示样本数, C表示通道数, T表示总帧数,V表示节点数,M表示视频中的人数

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)    # permute函数可以对任意高维矩阵进行转置
        x = self.data_bn(x)  # contiguous一般与transpose，permute,view搭配使用
                             # 即使用transpose或permute进行维度变换后，调用contiguous，然后方可使用view对维度进行变形。
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).view(N * M, C, T, V)  # N*M,C,T,V

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.stja1(x)
        x = self.l6(x)
        x = self.stja2(x)
        x = self.l7(x)
        x = self.stja3(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)    # 全连接分类
