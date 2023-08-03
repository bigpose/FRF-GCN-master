# operator：xcl
# operating time：2023/3/28 8:56
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .activations import *


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


class unit_tcn(nn.Module):      
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
         #                      stride=(stride, 1), dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)    
        self.relu = nn.ReLU()                    
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
       # x = self.relu(x)
        return x

class unit_tcn1(nn.Module):       
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn1, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1)) 

        self.bn = nn.BatchNorm2d(out_channels)    
        self.relu = nn.ReLU()                    
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class MultiScale_Temporal_SL(nn.Module): 
    def __init__(self, in_channels, kernel_size=5, expand_ratio=0.25, stride=1, dilations=[1, 3], residual=True, residual_kernel_size=1):
        super().__init__()
        inner_channel = int(in_channels * expand_ratio)
        compress_channel = int(in_channels/4)
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, inner_channel, 1, bias=True),
            nn.BatchNorm2d(inner_channel),
        )
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                TemporalConv(inner_channel, inner_channel, kernel_size=kernel_size, stride=stride, dilation=dilation),
                nn.BatchNorm2d(inner_channel),
                nn.ReLU(inplace=True),
                TemporalConv(inner_channel, compress_channel, kernel_size=1, stride=1, dilation=1),
            )
            for dilation in dilations
        ])

        self.branches.append(nn.Sequential(
            nn.Conv2d(inner_channel, compress_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(compress_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(compress_channel)  
        ))

        self.branches.append(nn.Sequential(
           nn.Conv2d(inner_channel, compress_channel, kernel_size=1, padding=0, stride=(stride,1)),
           nn.BatchNorm2d(compress_channel)
        ))
        
        # self.branches.append(nn.Sequential(
        #     nn.Conv2d(inner_channel, compress_channel, kernel_size=1, padding=0),
        #     nn.BatchNorm2d(compress_channel),
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
        #     nn.BatchNorm2d(compress_channel)  
        # ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, in_channels, kernel_size=residual_kernel_size, stride=stride)
        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        x = self.expand_conv(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            groups=16,
            bias=bias)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Temporal_SL(nn.Module):  
    def __init__(self, channel, temporal_window_size=9, bias=True, reduct_ratio=2, stride=1, residual=True, **kwargs):
        super(Temporal_SL, self).__init__()

        padding = (temporal_window_size - 1) // 2
        inner_channel = channel // reduct_ratio
        self.act = nn.Hardswish()
     
        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, (temporal_window_size,1), 1, (padding,0), groups=16, bias=bias),
            nn.BatchNorm2d(channel),
        )
        self.point_conv1 = nn.Sequential(
            nn.Conv2d(channel, inner_channel, 1, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv2 = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )
        self.depth_conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, (temporal_window_size,1), (stride,1), (padding,0), groups=16, bias=bias),
            nn.BatchNorm2d(channel),
        )

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            #self.residual = Temporal_SL_RESI(channel, channel, kernel_size=1, stride=stride)
            self.residual = nn.Sequential(
               nn.Conv2d(channel, channel, 1, (stride,1), bias=bias),
                nn.BatchNorm2d(channel),
            )


    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.depth_conv1(x))
        x = self.point_conv1(x)
        x = self.act(self.point_conv2(x))
        x = self.depth_conv2(x)
        return x + res

class Zero_Layer(nn.Module):
    def __init__(self):
        super(Zero_Layer, self).__init__()

    def forward(self, x):
        return 0

class Temporal_SL_dilation2(nn.Module): 
    def __init__(self, in_channels, temporal_window_size=5, bias=True, reduct_ratio=2, stride=1,
                 residual_kernel_size=1, residual=True, dilation=2, **kwargs):
        super(Temporal_SL_dilation2, self).__init__()

        padding = (temporal_window_size - 1) // 2
        pad1 = (temporal_window_size + (temporal_window_size-1) * (dilation-1) - 1) // 2
        pad2 = (residual_kernel_size - 1) // 2
        inner_channel = in_channels // reduct_ratio
        self.act = nn.Hardswish()
     
        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (temporal_window_size, 1), 1, (padding, 0), groups=16, bias=bias),
            nn.BatchNorm2d(in_channels),
        )
        self.point_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inner_channel, 1, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv2 = nn.Sequential(
            nn.Conv2d(inner_channel, in_channels, 1, bias=bias),
            nn.BatchNorm2d(in_channels),
        )
        self.depth_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (temporal_window_size, 1), stride=(stride, 1), padding=(pad1, 0),
                      dilation=(dilation, 1), groups=16, bias=bias),
            nn.BatchNorm2d(in_channels),
        )

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            # self.residual = TemporalConv(in_channels, in_channels, kernel_size=residual_kernel_size, stride=stride)
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, (residual_kernel_size, 1), stride=(stride, 1), padding=(pad2, 0),
                                  groups=16, bias=bias),
                nn.BatchNorm2d(in_channels),
            )
    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.depth_conv1(x))
        x = self.point_conv1(x)
        x = self.act(self.point_conv2(x))
        x = self.depth_conv2(x)
        return x + res

class Temporal_SL_dilation(nn.Module): 
    def __init__(self, in_channels, temporal_window_size=5, bias=True, reduct_ratio=2, stride=1,
                 residual_kernel_size=1, residual=True, dilation=2, **kwargs):
        super(Temporal_SL_dilation, self).__init__()

        padding = (temporal_window_size - 1) // 2 
        inner_channel = in_channels // reduct_ratio
        self.act = nn.Hardswish()
      
        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (temporal_window_size, 1), 1, (padding, 0), groups=16, bias=bias),
            nn.BatchNorm2d(in_channels),
        )
        self.point_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inner_channel, 1, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv2 = nn.Sequential(
            nn.Conv2d(inner_channel, in_channels, 1, bias=bias),
            nn.BatchNorm2d(in_channels),
        )
        self.depth_conv2 = TemporalConv(in_channels, in_channels, kernel_size=temporal_window_size,
                                        stride=stride, dilation=dilation)

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = TemporalConv(in_channels, in_channels, kernel_size=residual_kernel_size, stride=stride)
            # self.residual = nn.Sequential(
            #     nn.Conv2d(in_channels, in_channels, 1, (stride, 1), bias=bias),
            #     nn.BatchNorm2d(in_channels),
            # )
    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.depth_conv1(x))
        x = self.point_conv1(x)
        x = self.act(self.point_conv2(x))
        x = self.depth_conv2(x)
        return x + res

class unit_gcn(nn.Module):      # spatial GCN + bn + relu
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding     
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
     
        nn.init.constant_(self.PA, 1e-6)
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
        A = A + self.PA
        # A = self.PA.cuda(x.get_device())
        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T) 
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)        
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = Temporal_SL(out_channels, stride=stride)
        #self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        #if not residual:
         #   self.residual = lambda x: 0

        #elif (in_channels == out_channels) and (stride == 1):
         #   self.residual = lambda x: x
            # self.residual = unit_tcn(in_channels, in_channels, kernel_size=9, stride=stride, dilation=3)
        #else:
            #self.residual = TemporalConv(in_channels, out_channels, kernel_size=1, stride=stride)
            #self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
            #self.residual = Temporal_SL(in_channels, stride=stride)
            #self.residual = unit_tcn(in_channels, out_channels, kernel_size=9, stride=stride, dilation=3)
    def forward(self,x):
        x = self.relu(self.tcn1(self.gcn1(x)))
        return x

class TCN_GCN_unit1(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit1, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn1(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn1(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class STC_Att(nn.Module):
    def __init__(self, channel, reduct_ratio, bias=True, **kwargs):
        super(STC_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_c = SELayer(channel)

        self.add_attention = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.Hardswish(),
        )


    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)               
        x_v = x.mean(2, keepdims=True).transpose(2, 3)  
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2)) 
        x_t, x_v = torch.split(x_att, [T, V], dim=2)  
        x_t_att = self.conv_t(x_t).sigmoid()          
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att_f = x_t_att * x_v_att                    
        x_att_ff = x * x_att_f
        x_c_att = self.conv_c(x)                      
        
        x_att = x_att_ff + x_c_att
        # x_att = x_att_ff + x_c_att
        x_att = self.add_attention(x_att)
        return x_att

class STC_Att2(nn.Module):
    def __init__(self, channel, reduct_ratio, bias=True, **kwargs):
        super(STC_Att2, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_c = SELayer(channel)

        self.add_attention = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.Hardswish(),
        )

    def forward(self, x):
        x_t = x.mean(3, keepdims=True)                
        x_v = x.mean(2, keepdims=True)                 
        x_att_t = self.fcn(x_t)                        
        x_att_v = self.fcn(x_v)
        x_t_att = self.conv_t(x_att_t).sigmoid()       
        x_v_att = self.conv_v(x_att_v).sigmoid()
        x_att_f = x_t_att * x_v_att                   
        x_att_ff = x * x_att_f
        x_c_att = self.conv_c(x)                      
        x_att = x_att_ff + x_c_att                      
        x_att = self.add_attention(x_att)
        return x_att

class STC_Att3(nn.Module):
    def __init__(self, channel, reduct_ratio, bias=True, **kwargs):
        super(STC_Att3, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_c = nn.Conv2d(inner_channel, channel, kernel_size=1)

        self.add_attention = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.Hardswish(),
        )

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)                        
        x_v = x.mean(2, keepdims=True).transpose(2, 3)        
        x_c = x.mean(3, keepdims=True)
        x_c = x_c.mean(2, keepdims=True)                       
        x_att = self.fcn(torch.cat([x_t, x_v, x_c], dim=2))    
        x_t, x_v, x_c = torch.split(x_att, [T, V, 1], dim=2)   
        x_t_att = self.conv_t(x_t).sigmoid()                    
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_c_att = self.conv_c(x_c).sigmoid()
        x_att_f = x_t_att * x_v_att * x_c_att
        x_att = x_att_f * x
        #x_att2 = (x_att1 + x)
        x_att = self.add_attention(x_att)
        return x_att

class SELayer(nn.Module):                           
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        N, C, _, _ = x.size()
        x_c1 = self.avg_pool(x).view(N, C)           
        x_c2 = self.fc(x_c1).view(N, C, 1, 1)        
        return x * x_c2.expand_as(x)

class ST_Joint_Att(nn.Module):
    def __init__(self, channel, reduct_ratio, bias=True, **kwargs):
        super(ST_Joint_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)                  
        x_v = x.mean(2, keepdims=True).transpose(2, 3) 
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))  
        x_t, x_v = torch.split(x_att, [T, V], dim=2)   
        x_t_att = self.conv_t(x_t).sigmoid()          
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att                      
        return x_att


class Model(nn.Module):
    def __init__(self, num_class=120, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=6):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.stja1 = STC_Att3(128, 4)
        #self.stja2 = STC_Att(64, 4)
        self.stja2 = STC_Att3(128, 4)
        #self.stja4 = STC_Att(64, 4)
        self.stja3 = STC_Att3(128, 4)
        self.stja5 = STC_Att(256, 4)
        self.stja6 = STC_Att(256, 4)
        self.stja7 = STC_Att(256, 4)

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
        #if drop_out:
         #   self.drop_out = nn.Dropout(drop_out)
        #else:
        #    self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()    

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)    
        x = self.data_bn(x)  
                            
        x = x.view(N, M, V, C, T).permute(0, 3, 4, 2, 1)  # N,C,T,V,M
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        #x = self.stja4(x)
        x = self.l5(x)
        x = self.stja1(x)
        x = self.l6(x)
        x = self.stja2(x)
        x = self.l7(x)
        x = self.stja3(x)
        x = self.l8(x)
        #x = self.stja5(x)
        #x = self.stja2(x)
        x = self.l9(x)
        #x = self.stja6(x)
        x = self.l10(x)
        #x = self.stja7(x)


        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        # x2, _ = torch.max(x, 3, keepdim=False)
        # x3, _ = torch.max(x2, 1, keepdim=False)
        # x = torch.cat((x1, x3), dim=-2)
        return self.fc(x)   
