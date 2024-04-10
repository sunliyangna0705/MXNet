import os
import os.path
import argparse
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import h5py
import PIL
from PIL import Image
from network.dusrnet import DuSRNet
from torch.utils.data import DataLoader
# from deeplesion.build_gemotry import initialization, build_gemotry
from fastmri.data import transforms as T
from dataset.DataSet import DuDataset
from dataset.DataSet import CropDataset
from utils.my_loss import MyLoss
from utils.metric import complex_psnr
from utils.metric import complex_ssim
from utils.metric import nrmse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

os.environ['CUDA_VISIBLE_DEVICES'] = 'cpu'
parser = argparse.ArgumentParser(description="YU_Test")
# parser.add_argument("--model_dir", type=str, default="model", help='path to model and log files')
parser.add_argument("--model_dir", type=str, default="models", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="E:\\data\\ADMMCSNet\\dataset\\dataset\\test", help='path to testing data')
parser.add_argument("--write_file", type=str, default="OutFile.h5", help='write file for h5')
parser.add_argument("--use_GPU", type=bool, default=False, help='use GPU or not')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--save_img_dir', default='../test_imgs/', help='saving temp images')
# parser.add_argument("--save_path", type=str, default="test_result", help='path to save data')
parser.add_argument("--save_path", type=str, default="test_result/imgs", help='path to save data')
parser.add_argument('--num_channel', type=int, default=128, help='the number of dual channels')
parser.add_argument('--patchSize', type=int, default=255, help='the height / width of the input image to network')
parser.add_argument('--T', type=int, default=2, help='the number of ResBlocks in every ProxNet')
parser.add_argument('--S', type=int, default=10, help='the number of total iterative stages')
parser.add_argument('--fai', type=float, default=10, help='initialization for stepsize eta1')
parser.add_argument('--batchSize', type=int, default=1, help='batch size for test')
parser.add_argument('--batchnum', type=int, default=50,
                    help='batchsize*batchnum=1000 for randomly selecting 1000 imag pairs at every iteration')

opt = parser.parse_args()

if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = 'cpu'


def image_get_minmax():
    return 0.0, 1.0


def proj_get_minmax():
    return 0.0, 4.0


def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)  # 将数据处理成0-0.5范围
    data = (data - data_min) / (data_max - data_min)
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)), 0)
    return data


def print_network(name, net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('name={:s}, Total number={:d}'.format(name, num_params))


def zero_filling(datasets):
    data_loader = DataLoader(datasets, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers),
                             pin_memory=True)

    test_err = 0
    test_psnr = 0
    test_batches = 0
    test_ssim = 0
    test_nrmse = 0

    PSNR = []
    SSIM = []
    NRMSE = []


    for ii, data in enumerate(data_loader):
        print('saving images : [ %d / %d ]' % (ii, len(data_loader)))
        # gc.collect()
        with torch.no_grad():
            HRI, LRI, HRsp, LRsp, Mask = [x for x in data]

            X1 = LRI[0, 0, :, :].cpu()
            Xt = (T.tensor_to_complex_np(X1))

            HR1 = HRI[0, 0, :, :].cpu()
            HRt = (T.tensor_to_complex_np(HR1))

            # test_loss_normal = criterion(, HRI)
            # test_err += test_loss_normal.item()
            test_batches += 1
            # test_psnr_value = complex_psnr(abs(test_X).cpu().numpy(), abs(HRI).cpu().numpy(),
            #                                peak='normalized')
            test_psnr_value = complex_psnr(abs(HRt), abs(Xt),
                                           peak='normalized')
            PSNR.append(test_psnr_value)
            test_psnr += test_psnr_value

            test_ssim_value = complex_ssim(abs(HRt), abs(Xt))
            SSIM.append(test_ssim_value)
            test_ssim += test_ssim_value

            test_nrmse_value = nrmse(abs(HRt), abs(Xt))
            NRMSE.append(test_nrmse_value)
            test_nrmse += test_nrmse_value


    test_err /= test_batches
    test_psnr /= test_batches
    test_ssim /= test_batches
    test_nrmse /= test_batches

    print("test_loss ", test_err)
    print("test_psnr ", test_psnr, "mean: ", np.mean(PSNR), "var: ", np.std(PSNR))
    print("test_ssim ", test_ssim, "mean: ", np.mean(SSIM), "var: ", np.std(SSIM))
    print("test_nrmse ", test_nrmse, "mean: ", np.mean(NRMSE), "var: ", np.std(NRMSE))

    print("Successful Save file")


if __name__ == "__main__":
    # main()
    test_dataset = DuDataset(opt.data_path, opt.patchSize, 50)
    criterion = MyLoss()
    # net = DuSRNet(opt).cuda()
    zero_filling(test_dataset)
