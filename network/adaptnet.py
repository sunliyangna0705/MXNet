"""
Introduction
# 2022.3.1  by linasun
# 将其修改成fastmri形式
# 2022.5.16 修改mask为优化形式
# 2022.6.2 去掉mask优化，直接用列最大来找到mask
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
import MyLib as ML

sys.path.append(r'/data/sunlina/Net/DuSR-v5/network/')
# import MyLib as ML
if version.parse(torch.__version__) >= version.parse("1.7.0"):
    import torch.fft  # type: ignore

filter = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) / 9  # initialization 初始化的滤波器,shape（3,3）
filter = filter.unsqueeze(dim=0).unsqueeze(dim=0)  # shape(1,1,3,3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class adaptnet(nn.Module):
    def __init__(self, args):
        super(adaptnet, self).__init__()
        self.S = args.S  # Stage number S includes the initialization process
        self.iter = self.S - 1  # not include the initialization process
        # self.iter = 1
        self.num_u = args.num_channel + 1  # concat extra 1 term
        self.num_f = args.num_channel + 2  # concat extra 2 terms
        self.T = args.T
        self.fai = args.fai
        self.fai2 = nn.Parameter(torch.ones(args.S) * args.fai, requires_grad=True)
        self.img_path = args.save_img_dir
        self.batch = args.batchSize
        self.channel = args.num_channel
        self.sigma = 0.1
        self.frc = args.acc
        # proxNet for initialization
        self.proxNet_X0 = ProjnetX(args.num_channel + 2, self.T)
        self.proxNet_M0 = ProjnetM(args.num_mask_channel , self.T)

        # get width and height of figures
        self.W = args.w
        self.H = args.h

        # proxNet for iterative process
        self.proxNet_Xall = self.make_Xnet(self.S, args.num_channel + 2, self.T)
        # self.proxNet_Mall = self.make_Mnet(self.S, args.num_mask_channel +1 , self.T)

        # 初始化时使用
        self.CX_const = filter.expand(args.num_channel, 2, -1, -1).clone()
        self.CX = nn.Parameter(self.CX_const, requires_grad=True)  # 这是最后的filter

        # self.bn = nn.BatchNorm2d(1)
        # self.CM_const = filter.expand(args.num_mask_channel, 1, -1, -1).clone()
        # self.CM = nn.Parameter(self.CM_const, requires_grad=True)  # 这是最后的filter

    def forward(self, X, Z, LI, Y, epoch):
        # Y  = LRsp    Z = HRsp
        # LI = LRI     X = HRI
        ListX = []  # saving the reconstructed figure
        ListZ = []  # saving the reconstructed sinogram
        save_mask_dir = '/data/sunlina/models/DuSR-mask-learning/test_result'
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
        X0 = X0_ini[:, :2, :, :]
        XZ = X0_ini[:, 2:, :, :]
        X = chan_complex_to_last_dim(X0)
        _,_,h,w,_ = Y.shape
        # mask = np.zeros((h,w))
        # a1, b1, c1, d1 = mask.shape
        maskLine = sum(Y[0, 0, :, :,0])
        value, idx = torch.sort(torch.abs(maskLine), descending=True)  # descending为alse，升序，为True，降序
        idx1 = idx[:int(self.frc * w)]
        idx0 = idx[int(self.frc * w):]
        maskLine[idx1] = 1
        maskLine[idx0] = 0
        mask = maskLine.reshape(1, 1, 1, w)
        mask = mask.repeat((1,1,h,1))
        ML.imwrite(mask[0, 0, :, :].cpu().detach().numpy(), os.path.join(save_mask_dir, 'mask_temp_i.png' ))


        for i in range(self.iter):
            # i st iteration: Updating Z
            Z2 = fastmri.fft2c(X)  # (3,1,320,320,2)
            # mask optimation 2022.5.16
            # maskUP = self.sigma * complex_conjugate_square(Y)
            # maskDown = complex_conjugate_square(Z2-Y) + maskUP
            # maskini = maskUP / maskDown
            # mask00 = F.conv2d(maskini, self.CM, padding=1)
            # maskCat = torch.cat((maskini, mask00), dim=1)
            # mask = self.proxNet_Mall[0](maskCat)
            # mask = mask[:,0,:,:].unsqueeze_(1)
            # a1,b1,c1,d1 = mask.shape
            # # print("mask shape ",a1,b1,c1,d1 )
            # # maskLine = mask.reshape(1,a1*b1*c1*d1)
            # maskLine = sum(mask[0,0,:,:])
            #
            # value, idx = torch.sort(torch.abs(maskLine), descending=True)  # descending为alse，升序，为True，降序
            # idx1 = idx[:int(self.frc*256)]
            # idx0 = idx[int(self.frc*256):]
            # maskLine[idx1] = 1
            # maskLine[idx0] = 0
            # mask = maskLine.reshape(a1,b1,1,d1)
            # mask = mask.repeat((1,1,c1,1))
            # # mask_update = torch.stack((mask,mask),-1)

            # Xt = mask[0, 0, :, :].cpu().detach().numpy()
            # ML.imwrite(Xt, os.path.join(self.img_path, 'MaskInit.png'))

            # if epoch % 200 == 0:
            #     print("----------------saving figure--------%d--------" % i)
            #     Xt = mask[0,0,:,:].cpu().detach().numpy()
            #     ML.imwrite(Xt, os.path.join(self.img_path, 'tempMask%d_epoch_%d.png' % (i, epoch)))

            MaskY = self.fai2[-1] * torch.stack((mask, mask), -1) * Y
            Z = (Z2 + MaskY) / (1 + self.fai2[-1] * torch.stack((mask, mask), -1))
            # i st iteration: Updating X
            IFTZ = fastmri.ifft2c(Z)
            IFdim = complex_to_chan_dim(IFTZ)
            IFTZ1 = torch.cat((IFdim, XZ), dim=1)
            outX = self.proxNet_Xall[0](IFTZ1)
            X = chan_complex_to_last_dim(outX[:, :2, :, :])
            XZ = outX[:, 2:, :, :]

            ListX.append(X)
            ListZ.append(Z)

        return ListX, ListZ,mask

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
