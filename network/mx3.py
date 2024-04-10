"""
MXNet 2024.3.1 by linasun

Parameters:
- low_resolution_input: Low-resolution image input (Y).
- high_resolution_sinogram: High-resolution sinogram (Z).
- low_resolution_sinogram: Low-resolution sinogram (LI).
- high_resolution_input: High-resolution image input (X).
- epoch: Current epoch number.

Returns:
- list_reconstructed_images: List of reconstructed high-resolution images over iterations.
- list_reconstructed_sinograms: List of reconstructed high-resolution sinograms over iterations.
- mask: Final computed mask.
"""
import torch
import torch.nn as nn
import torch.nn.functional as  F
import fastmri
from packaging import version
import sys
sys.path.append(r'/data/sunlina/Net/DuSR-v5/network/')
if version.parse(torch.__version__) >= version.parse("1.7.0"):
    import torch.fft

filter = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) / 9
filter = filter.unsqueeze(dim=0).unsqueeze(dim=0)  # shape(1,1,3,3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class mxnet(nn.Module):
    def __init__(self, args):
        super(mxnet, self).__init__()
        self.S = args.S  # Stage number S includes the initialization process
        self.T = args.T
        self.fai = args.fai
        self.fai2 = nn.Parameter(torch.ones(args.S) * args.fai, requires_grad=True)
        self.img_path = args.save_img_dir
        self.batch = args.batchSize
        self.channel = args.num_channel
        self.sigma = nn.Parameter(torch.ones(args.S) * 0.1, requires_grad=True)
        self.proxNet_X0 = ProjnetX(args.num_channel + 2, self.T)
        self.proxNet_M0 = ProjnetM(args.num_mask_channel , self.T)
        # proxNet for iterative process
        self.proxNet_Xall = self.make_Xnet(self.S, args.num_channel + 2, self.T)
        self.proxNet_Mall = self.make_Mnet(self.S, args.num_mask_channel +1 , self.T)
        # initial network
        self.CX_const = filter.expand(args.num_channel, 2, -1, -1).clone()
        self.CX = nn.Parameter(self.CX_const, requires_grad=True)
        self.CM_const = filter.expand(args.num_mask_channel, 1, -1, -1).clone()
        self.CM = nn.Parameter(self.CM_const, requires_grad=True)  # 这是最后的filter

    def forward(self, Y):

        ListX = []  # saving the reconstructed figure
        ListZ = []  # saving the reconstructed sinogram

        # Initial Z0
        Z0 = Y
        image_data = fastmri.ifft2c(Z0)
        image_data2 = complex_to_chan_dim(image_data)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_data2 = image_data2.to(device)
        cov_data = F.conv2d(image_data2, self.CX, padding=1)  # (3,32,320,320)
        cov_data = cov_data.to(device)
        cat_data = torch.cat((image_data2, cov_data), dim=1)  # (3,34,320,320)
        ini_X0 = self.proxNet_Xall[0](cat_data)

        X0 = ini_X0[:, :2, :, :]
        XZ = ini_X0[:, 2:, :, :]
        X = chan_complex_to_last_dim(X0)

        for i in range(self.S):
            # i st iteration: Updating Z
            Zi = fastmri.fft2c(X)
            n_mask = self.sigma[-1] * complex_conjugate_square(Y) +  complex_conj_multiply(Y, Zi)
            n_mask_sum = torch.sum(n_mask, dim=-2, keepdim=True)

            d_mask = self.sigma[-1] * complex_conjugate_square(Y) +complex_conjugate_square(Zi)
            d_mask_sum = torch.sum(d_mask, dim=-2, keepdim=True)
            init_mask = n_mask_sum / d_mask_sum
            init_mask2 = init_mask.repeat(1, 1, n_mask.shape[-2], 1)
            cov_mask = F.conv2d(init_mask2, self.CM, padding=1)
            cat_mask = torch.cat((init_mask2, cov_mask), dim=1)
            prox_mask = self.proxNet_Mall[0](cat_mask)
            mask = prox_mask[:, 0, :, :].unsqueeze_(1)


            maskY = self.fai2[-1] * torch.stack((mask, mask), -1) * Y
            Z = (Zi + maskY) / (1 + self.fai2[-1] * torch.stack((mask, mask), -1))

            # i st iteration: Updating X
            pse_X = fastmri.ifft2c(Z)
            pse_X2 = complex_to_chan_dim(pse_X)
            cat_X = torch.cat((pse_X2, XZ), dim=1)
            out_X = self.proxNet_Xall[0](cat_X)
            X = chan_complex_to_last_dim(out_X[:, :2, :, :])
            XZ = out_X[:, 2:, :, :]

            ListX.append(X)
            ListZ.append(Z)


        return ListX, ListZ, mask


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
        # self.sigmoid = nn.Sigmoid(inplace=True)

    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(
                nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              # nn.BatchNorm2d(self.channels),
                              nn.ReLU(),
                              nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              # nn.BatchNorm2d(self.channels),
                              # nn.Sigmoid(),
                              ))

        # return nn.Sequential(*layers)
        return nn.ModuleList(layers)

    def forward(self, input):
        X = input
        for i in range(self.T):
            #  X = F.relu(X + self.layer[i](X))
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
    return x[:,:,:,:,0]**2 + x[:,:,:,:,1]**2

def complex_conj_multiply(x: torch.Tensor,y: torch.Tensor) -> torch.Tensor:

    return 2*(x[:,:,:,:,0]*y[:,:,:,:,0]+x[:,:,:,:,1]*y[:,:,:,:,1])
