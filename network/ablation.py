"""
Introduction
# 2022.3.1  by linasun
# 将其修改成fastmri形式
# 2022.5.16 修改mask为优化形式
# 去掉acc参数，按照0-1规划来求解mask优化
# 2022.11.16
# 消融实验

"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as  F
from fastmri.data import transforms as T
import fastmri
from packaging import version
from matplotlib import pyplot as plt
import numpy as np
import sys
from scipy.io import loadmat
import MyLib as ML
sys.path.append(r'/data/sunlina/Net/DuSR-v5/network/')
# import MyLib as ML
if version.parse(torch.__version__) >= version.parse("1.7.0"):
    import torch.fft  # type: ignore

filter = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) / 9  # initialization 初始化的滤波器,shape（3,3）
filter = filter.unsqueeze(dim=0).unsqueeze(dim=0)  # shape(1,1,3,3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import h5py
def write_to_h5(np_array, file_name, dict_name):
    f = h5py.File(file_name, 'w')
    f[dict_name] = np_array
    f.close()


class ablationnet(nn.Module):
    def __init__(self, args):
        super(ablationnet, self).__init__()
        self.S = args.S  # Stage number S includes the initialization process
        self.iter = self.S - 1  # not include the initialization process，the number of Stage
        self.T = args.T
        self.fai = args.fai
        self.fai2 = nn.Parameter(torch.ones(args.S) * args.fai, requires_grad=True)
        self.img_path = args.save_img_dir
        self.batch = args.batchSize
        self.channel = args.num_channel
        # self.sigma = 0.1
        self.sigma = nn.Parameter(torch.ones(args.S) * 0.1, requires_grad=True)
        # self.frc = args.acc
        # proxNet for initialization
        self.proxNet_X0 = ProjnetX(args.num_channel + 2, self.T)
        self.proxNet_M0 = ProjnetM(args.num_mask_channel , self.T)

        # proxNet for iterative process
        self.proxNet_Xall = self.make_Xnet(self.S, args.num_channel + 2, self.T)
        self.proxNet_Mall = self.make_Mnet(self.S, args.num_mask_channel +1 , self.T)

        # 初始化时使用
        self.CX_const = filter.expand(args.num_channel, 2, -1, -1).clone()
        self.CX = nn.Parameter(self.CX_const, requires_grad=True)  # 这是最后的filter

        # self.bn = nn.BatchNorm2d(1)
        self.CM_const = filter.expand(args.num_mask_channel, 1, -1, -1).clone()
        self.CM = nn.Parameter(self.CM_const, requires_grad=True)  # 这是最后的filter

    def forward(self, X, Z, LI, Y, epoch):
        # Y  = LRsp    Z = HRsp
        # LI = LRI     X = HRI
        ListX = []  # saving the reconstructed figure
        ListZ = []  # saving the reconstructed sinogram
        save_mask_dir = '/data/sunlina/models/DuSR-mask-learning-Node11/test_result'
       # Initial Z0
        Z0 = Y
        # 1st iteration: Updating X0, X0-->X1
        XZi = fastmri.ifft2c(Z0)  # (3,1,320,320,2)
        XZi0 = complex_to_chan_dim(XZi)  # (3,2,320,320)
        XZi0 = XZi0.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        XZ00 = F.conv2d(XZi0, self.CX, padding=1)  # (3,32,320,320)
        XZi = XZi.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        XZ00 = XZ00.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        XZ = torch.cat((XZi0, XZ00), dim=1)  # (3,34,320,320)
        X0_ini = self.proxNet_Xall[0](XZ)
        # 将实部虚部两维度拆出来
        X0 = X0_ini[:, :2, :, :]  # (3,2,320,320)
        XZ = X0_ini[:, 2:, :, :]
        X = chan_complex_to_last_dim(X0)
        # print("X shape: ", X.shape)

        for i in range(self.iter):
            # i st iteration: Updating Z
            Z2 = fastmri.fft2c(X)
            # print("Z2 shape: ",Z2.shape)

            # mask optimation 2022.6.20 推导
            # 消融实验
            # maskUP = self.sigma[-1] * complex_conjugate_square(Y)+self.sigma[-1]*complex_conj_multiply(Y,Z2)
            # maskUPsum = torch.sum(maskUP,dim=-2,keepdim=True)
            # # maskDown = complex_conjugate_square(Z2-Y) + maskUP
            # maskDown = complex_conjugate_square(Z2) + self.sigma[-1] * complex_conjugate_square(Y)
            # maskDownsum = torch.sum(maskDown,dim=-2,keepdim=True)
            # maskini0 = maskUPsum / maskDownsum
            # one = torch.ones_like(maskini0)
            # zero = torch.zeros_like(maskini0)
            # maskini01 = torch.where(maskini0 > 0.5, one, zero)
            # maskini = maskini0.repeat(1,1,maskUP.shape[-2],1)
            # mask00 = F.conv2d(maskini, self.CM, padding=1)
            # maskCat = torch.cat((maskini, mask00), dim=1)
            ## 消融实验
            Y2 = complex_to_chan_dim(Y)
            maskini = torch.abs(Y2)/torch.norm(Y2)
            mask = self.proxNet_Mall[0](maskini)
            mask = mask[:, 0, :, :].unsqueeze_(1)
            # maskCat = torch.cat((maskini, maskini), dim=1)
            # mask = self.proxNet_Mall[0](maskCat)
            # 消融实验截至
            # mask = self.proxNet_Mall[0](complex_to_chan_dim(Y))

            # one = torch.ones_like(mask)
            # zero = torch.zeros_like(mask)
            # mask = torch.maximum(mask, one)
            # mask = torch.minimum(mask, zero)

            ML.imwrite(((mask[0, 0, :, :].cpu())),
                       os.path.join(save_mask_dir, 'mask_temp_%i.png' % (i)))


            MaskY = self.fai2[-1] * torch.stack((mask, mask), -1) * Y
            Z = (Z2 + MaskY) / (1 + self.fai2[-1] * torch.stack((mask, mask), -1))
            # i st iteration: Updating X
            IFTZ = fastmri.ifft2c(Z)
            IFdim = complex_to_chan_dim(IFTZ)
            IFTZ1 = torch.cat((IFdim, XZ), dim=1)
            outX = self.proxNet_Xall[0](IFTZ1)
            X = chan_complex_to_last_dim(outX[:, :2, :, :])
            XZ = outX[:, 2:, :, :]
            ML.imwrite(abs(T.tensor_to_complex_np(X[0,0,:,:].cpu())), os.path.join(save_mask_dir, 'X_temp_%i.png' % (i)))
            write_to_h5(abs(T.tensor_to_complex_np(X[0,0,:,:].cpu())), os.path.join(save_mask_dir, 'CX_%i.h5' % (i)),
                        'CX')
            ListX.append(X)
            ListZ.append(Z)
        one = torch.ones_like(mask)
        zero = torch.zeros_like(mask)
        OutMask = torch.where(mask > 0.5, one, zero)
        ML.imwrite(((mask[0, 0, :, :].cpu())),
                   os.path.join(save_mask_dir, 'mask_out.png' ))


        return ListX, ListZ,OutMask

    def make_Xnet(self, iters, channel, T):  #
        layers = []
        for i in range(iters):
            layers.append(ProjnetX(channel, T))
        return nn.Sequential(*layers)

    def make_Mnet(self, iters, channel, T):  #
        layers = []
        for i in range(iters):
            layers.append(ProjnetM(channel, T))
        return nn.Sequential(*layers)


# proxNet_X
class ProjnetX(nn.Module):
    def __init__(self, channel, T):
        super(ProjnetX, self).__init__()
        self.channels = channel  # channels = 32
        self.T = T  # T = 4
        self.layer = self.make_resblock(self.T)

    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(
                nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              # nn.BatchNorm2d(self.channels),
                              nn.ReLU(),
                              nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              # nn.BatchNorm2d(self.channels),
                              ))
        return nn.Sequential(*layers)

    def forward(self, input):
        X = input
        for i in range(self.T):
            #  X = F.relu(X + self.layer[i](X))
            X = (X + self.layer[i](X))

        return X


# proxNet_M
class ProjnetM(nn.Module):
    def __init__(self, channel, T):
        super(ProjnetM, self).__init__()
        self.channels = channel  # channels = 32
        self.T = T  # T = 4
        self.layer = self.make_resblock(self.T)

    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(
                nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              # nn.BatchNorm2d(self.channels),
                              nn.ReLU(),
                              nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              # nn.BatchNorm2d(self.channels),
                              ))
        return nn.Sequential(*layers)

    def forward(self, input):
        X = input
        for i in range(self.T):
            #  X = F.relu(X + self.layer[i](X))
            X = (X + self.layer[i](X))

        return X


def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w, two = x.shape
    assert two == 2
    return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)


def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
    b, c2, h, w = x.shape
    assert c2 % 2 == 0
    c = c2 // 2
    return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

def complex_conjugate_square(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w, two = x.shape
    # CojSqr = x[:,:,:,:,0]**2 + x[:,:,:,:,1]**2
    return x[:,:,:,:,0]**2 + x[:,:,:,:,1]**2

def complex_conj_multiply(x: torch.Tensor,y: torch.Tensor) -> torch.Tensor:

    return 2*(x[:,:,:,:,0]*y[:,:,:,:,0]+x[:,:,:,:,1]*y[:,:,:,:,1])
