import torch
import torch.nn as nn
import torch.nn.functional as  F
from fastmri.data import transforms as T
import fastmri
from packaging import version
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.append(r'/data/Net/network/')
# torch.autograd.set_detect_anomaly(True)

if version.parse(torch.__version__) >= version.parse("1.7.0"):
    import torch.fft  # type: ignore

filter = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) / 9
filter = filter.unsqueeze(dim=0).unsqueeze(dim=0)  # shape(1,1,3,3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class net(nn.Module):
    def __init__(self, args):
        super(net, self).__init__()
        self.S = args.S  # Stage number S includes the initialization process
        self.iter = self.S - 1  # not include the initialization process，the number of Stage
        self.T = args.T
        self.fai = args.fai
        self.fai2 = nn.Parameter(torch.ones(args.S) * args.fai, requires_grad=True)
        self.img_path = args.save_img_dir
        self.batch = args.batchSize
        self.channel = args.num_channel
        self.sigma = nn.Parameter(torch.ones(args.S) * 0.1, requires_grad=True)
        # proxNet for initialization
        self.proxNet_X0 = ProjnetX(args.num_channel + 2, self.T)
        self.proxNet_M0 = ProjnetM(args.num_mask_channel , self.T)

        # proxNet for iterative process
        self.proxNet_Xall = self.make_Xnet(self.S, args.num_channel + 2, self.T)
        self.proxNet_Mall = self.make_Mnet(self.S, args.num_mask_channel +1 , self.T)

        self.CX_const = filter.expand(args.num_channel, 2, -1, -1).clone()
        self.CX = nn.Parameter(self.CX_const, requires_grad=True)

        # self.bn = nn.BatchNorm2d(1)
        self.CM_const = filter.expand(args.num_mask_channel, 1, -1, -1).clone()
        self.CM = nn.Parameter(self.CM_const, requires_grad=True)

    def forward(self, Y):


        reconstructed_image_list = []   # saving the reconstructed figure
        reconstructed_k_space_list = [] # saving the reconstructed k-space


        Z0 = Y

        # 1st iteration: Updating X0, X0-->X1
        XZi = fastmri.ifft2c(Z0)
        XZi_reshape = complex_to_chan_dim(XZi)
        XZi_reshape = XZi_reshape.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        XZ_cov = F.conv2d(XZi_reshape, self.CX, padding=1)
        XZi = XZi.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        XZ_cov = XZ_cov.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        XZ = torch.cat((XZi_reshape, XZ_cov), dim=1)
        X0_ini = self.proxNet_Xall[0](XZ)

        X0 = X0_ini[:, :2, :, :]
        XZ = X0_ini[:, 2:, :, :]
        X = chan_complex_to_last_dim(X0)


        for i in range(self.iter):
            # i st iteration: Updating Z
            Z2 = fastmri.fft2c(X)

            mask_numerator = self.sigma[-1] * complex_conjugate_square(Y)+self.sigma[-1]*complex_conj_multiply(Y,Z2)
            mask_numerator_sum = torch.sum(mask_numerator,dim=-2,keepdim=True)
            # mask_denominator = complex_conjugate_square(Z2-Y) + mask_numerator
            mask_denominator = complex_conjugate_square(Z2) + self.sigma[-1] * complex_conjugate_square(Y)
            mask_denominator_sum = torch.sum(mask_denominator,dim=-2,keepdim=True)
            initial_mask = mask_numerator_sum / mask_denominator_sum


            initial_mask = initial_mask.repeat(1,1,mask_numerator.shape[-2],1)
            mask_cov = F.conv2d(initial_mask, self.CM, padding=1)
            mask_concat = torch.cat((initial_mask, mask_cov), dim=1)
            mask = self.proxNet_Mall[0](mask_concat)
            mask = mask[:,0,:,:].unsqueeze_(1)


            MaskY = self.fai2[-1] * torch.stack((mask, mask), -1) * Y
            Z2 = fastmri.fft2c(X)
            Z = (Z2 + MaskY) / (1 + self.fai2[-1] * torch.stack((mask, mask), -1))

            # i st iteration: Updating X
            ifft_Z = fastmri.ifft2c(Z)
            ifft_Z_reshape = complex_to_chan_dim(ifft_Z)
            ifft_Z_concat = torch.cat((ifft_Z_reshape, XZ), dim=1)
            outX = self.proxNet_Xall[0](ifft_Z_concat)
            X = chan_complex_to_last_dim(outX[:, :2, :, :])
            XZ = outX[:, 2:, :, :]

            reconstructed_image_list.append(X)
            reconstructed_k_space_list.append(Z)
        one = torch.ones_like(mask)
        zero = torch.zeros_like(mask)
        OutMask = torch.where(mask > 0.5, one, zero)

        return reconstructed_image_list, reconstructed_k_space_list,OutMask

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
        # self.sigmoid = nn.Sigmoid()

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
        # output = self.sigmoid(X)

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
