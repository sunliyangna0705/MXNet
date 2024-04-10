import torch
import torch.nn as nn
import torch.nn.functional as F
import fastmri
from packaging import version
import h5py
import sys

sys.path.append('/data/Net/MXNet/network/')

if version.parse(torch.__version__) >= version.parse("1.7.0"):
    import torch.fft

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simplified filter initialization directly on the device
filter = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], device=device) / 9
filter = filter.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 3, 3)

# Function to write numpy array to h5 file
def write_to_h5(np_array, file_name, dict_name):
    with h5py.File(file_name, 'w') as f:
        f[dict_name] = np_array

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

    def forward(self, low_resolution_input, high_resolution_sinogram, low_resolution_sinogram, high_resolution_input,
                epoch):
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
        # Initialize lists for saving reconstructed images and sinograms
        list_reconstructed_images = []
        list_reconstructed_sinograms = []

        # Initialize Z0 from low-resolution input
        initial_sinogram = low_resolution_input

        # First iteration to update from initial state
        image_data = fastmri.ifft2c(initial_sinogram)
        image_data_with_channels = complex_to_chan_dim(image_data)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image_data_with_channels = image_data_with_channels.to(device)
        convolved_data = F.conv2d(image_data_with_channels, self.CX, padding=1)
        image_space_data = image_data.to(device)
        convolved_data = convolved_data.to(device)
        concatenated_data = torch.cat((image_data_with_channels, convolved_data), dim=1)
        proximal_output = self.proxNet_Xall[0](concatenated_data)

        # Separate the proximal output into components
        reconstructed_image_space_data = proximal_output[:, :2, :, :]
        additional_data = proximal_output[:, 2:, :, :]
        final_image_output = chan_complex_to_last_dim(reconstructed_image_space_data)

        for i in range(self.iter):
            high_res_sinogram_updated = fastmri.fft2c(final_image_output)

            # Update M
            numerator_mask = self.sigma[-1] * complex_conjugate_square(low_resolution_input) + \
                             self.sigma[-1] * complex_conj_multiply(low_resolution_input, high_res_sinogram_updated)
            numerator_sum = torch.sum(numerator_mask, dim=-2, keepdim=True)
            denominator_mask = complex_conjugate_square(high_res_sinogram_updated) + \
                               self.sigma[-1] * complex_conjugate_square(low_resolution_input)
            denominator_sum = torch.sum(denominator_mask, dim=-2, keepdim=True)
            mask_initial = numerator_sum / denominator_sum

            # Adjust mask dimensions and apply convolution
            mask_adjusted = mask_initial.repeat(1, 1, numerator_mask.shape[-2], 1)
            convolved_mask = F.conv2d(mask_adjusted, self.CM, padding=1)
            mask_concatenated = torch.cat((mask_adjusted, convolved_mask), dim=1)
            mask_final = self.proxNet_Mall[0](mask_concatenated)

            # Updating Z
            mask_final = mask_final[:, 0, :, :].unsqueeze_(1)
            mask_applied_to_Y = self.fai2[-1] * torch.stack((mask_final, mask_final), -1) * low_resolution_input
            high_resolution_sinogram = (high_res_sinogram_updated + mask_applied_to_Y) / \
                                       (1 + self.fai2[-1] * torch.stack((mask_final, mask_final), -1))

            # Updateing X
            ifft_high_resolution_sinogram = fastmri.ifft2c(high_resolution_sinogram)
            ifft_high_res_sinogram_with_channels = complex_to_chan_dim(ifft_high_resolution_sinogram)
            concatenated_for_prox_net = torch.cat((ifft_high_res_sinogram_with_channels, additional_data), dim=1)
            prox_net_output = self.proxNet_Xall[0](concatenated_for_prox_net)
            final_image_output = chan_complex_to_last_dim(prox_net_output[:, :2, :, :])
            additional_data = prox_net_output[:, 2:, :, :]

            # Append results to lists
            list_reconstructed_images.append(final_image_output)
            list_reconstructed_sinograms.append(high_resolution_sinogram)

        return list_reconstructed_images, list_reconstructed_sinograms, mask_final

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

