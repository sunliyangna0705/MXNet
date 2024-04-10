import torch
import torch.nn as nn
import torch.nn.functional as  F
import fastmri
from packaging import version
import sys
import h5py
sys.path.append(r'/data/Net/MXNet/network/')

if version.parse(torch.__version__) >= version.parse("1.7.0"):
    import torch.fft  # type: ignore

filter = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) / 9  # initialization 初始化的滤波器,shape（3,3）
filter = filter.unsqueeze(dim=0).unsqueeze(dim=0)  # shape(1,1,3,3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def write_to_h5(np_array, file_name, dict_name):
    f = h5py.File(file_name, 'w')
    f[dict_name] = np_array
    f.close()
class mxnet(nn.Module):
    def __init__(self, args):
        super(mxnet, self).__init__()
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
        # get width and height of figures
        self.W = args.w
        self.H = args.h
        # proxNet for iterative process
        self.proxNet_Xall = self.make_Xnet(self.S, args.num_channel + 2, self.T)
        self.proxNet_Mall = self.make_Mnet(self.S, args.num_mask_channel +1 , self.T)

        self.CX_const = filter.expand(args.num_channel, 2, -1, -1).clone()
        self.CX = nn.Parameter(self.CX_const, requires_grad=True)  # 这是最后的filter
        self.CM_const = filter.expand(args.num_mask_channel, 1, -1, -1).clone()
        self.CM = nn.Parameter(self.CM_const, requires_grad=True)  # 这是最后的filter

    def forward(self, X, Z, LI, Y, epoch):
        # Y  = LRsp    Z = HRsp
        # LI = LRI     X = HRI
        ListX = []  # saving the reconstructed figure
        ListZ = []  # saving the reconstructed sinogram
       # Initial Z0
        Z0 = Y
        # 1st iteration: Updating X0, X0-->X1
        XZi = fastmri.ifft2c(Z0)
        XZi0 = complex_to_chan_dim(XZi)
        XZi0 = XZi0.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        XZ00 = F.conv2d(XZi0, self.CX, padding=1)
        XZi = XZi.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        XZ00 = XZ00.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        XZ = torch.cat((XZi0, XZ00), dim=1)
        X0_ini = self.proxNet_Xall[0](XZ)

        X0 = X0_ini[:, :2, :, :]
        XZ = X0_ini[:, 2:, :, :]
        X = chan_complex_to_last_dim(X0)

        for i in range(self.iter):
            # i st iteration: Updating Z
            Z2 = fastmri.fft2c(X)
            # Update M
            maskUP = self.sigma[-1] * complex_conjugate_square(Y)+self.sigma[-1]*complex_conj_multiply(Y,Z2)
            maskUPsum = torch.sum(maskUP,dim=-2,keepdim=True)
            # maskDown = complex_conjugate_square(Z2-Y) + maskUP
            maskDown = complex_conjugate_square(Z2) + self.sigma[-1] * complex_conjugate_square(Y)
            maskDownsum = torch.sum(maskDown,dim=-2,keepdim=True)
            maskini0 = maskUPsum / maskDownsum

            maskini = maskini0.repeat(1,1,maskUP.shape[-2],1)
            mask00 = F.conv2d(maskini, self.CM, padding=1)
            maskCat = torch.cat((maskini, mask00), dim=1)
            mask = self.proxNet_Mall[0](maskCat)

            mask = mask[:,0,:,:].unsqueeze_(1)
            MaskY = self.fai2[-1] * torch.stack((mask, mask), -1) * Y
            # Update Z
            Z = (Z2 + MaskY) / (1 + self.fai2[-1] * torch.stack((mask, mask), -1))

            # Update X
            IFTZ = fastmri.ifft2c(Z)
            IFdim = complex_to_chan_dim(IFTZ)
            IFTZ1 = torch.cat((IFdim, XZ), dim=1)
            outX = self.proxNet_Xall[0](IFTZ1)
            X = chan_complex_to_last_dim(outX[:, :2, :, :])
            XZ = outX[:, 2:, :, :]

            ListX.append(X)
            ListZ.append(Z)
        # one = torch.ones_like(mask)
        # zero = torch.zeros_like(mask)
        # OutMask = torch.where(mask > 0.5, one, zero)

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


class ProjnetX(nn.Module):
    def __init__(self, channel, T):
        super(ProjnetX, self).__init__()
        self.channels = channel
        self.T = T
        self.layer = self.make_resblock(self.T)

    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(
                nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              nn.ReLU(),
                              nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),

                              ))
        return nn.Sequential(*layers)

    def forward(self, input):
        X = input
        for i in range(self.T):
            X = (X + self.layer[i](X))

        return X



class ProjnetM(nn.Module):
    def __init__(self, channel, T):
        super(ProjnetM, self).__init__()
        self.channels = channel
        self.T = T
        self.layer = self.make_resblock(self.T)
        # self.sigmoid = nn.Sigmoid(inplace=True)

    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(
                nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              nn.ReLU(),
                              nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              ))

        return nn.ModuleList(layers)

    def forward(self, input):
        X = input
        for i in range(self.T):
            X = (X + self.layer[i](X))
        # Out =torch.sigmoid (X).clone()

        return X
torch.autograd.set_detect_anomaly(True)


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
