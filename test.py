import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path
import argparse
import numpy as np
import torch
from network.mxnet import net
from torch.utils.data import DataLoader
from fastmri.data import transforms as T
from dataset.DataSet import DataSet
from utils.visual import tensor2np
from utils.my_loss import MyLoss
from utils.metric import complex_psnr
from utils.metric import complex_ssim
from utils.metric import nrmse
import h5py

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description="YU_Test")
parser.add_argument("--model_dir", type=str, default="/models/DuSR-mask-learning-Node11/models/19Dec_035237pm_0.8acc_2000ep_10T_10S",
                    help='path to model and log files')
parser.add_argument("--data_path", type=str, default="/data/Admm-data/test", help='path to testing data')
parser.add_argument("--mask_dir_tst", type=str, default="/data/Admm-data/mask_40/mask0.4.mat", help='test mask data')
parser.add_argument("--write_file", type=str, default="OutFile.h5", help='write file for h5')
parser.add_argument("--use_GPU", type=bool, default=False, help='use GPU or not')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--save_img_dir', default='/models/DuSR-mask-learning/test_imgs/', help='saving temp images')
parser.add_argument("--save_path", type=str, default="/models/DuSR-mask-learning/test_result/imgs", help='path to save data')
parser.add_argument('--num_channel', type=int, default=64, help='the number of dual channels')
parser.add_argument('--num_mask_channel', type=int, default=1, help='the number of channel for mask')
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



def test_model(net, datasets):
    data_loader = DataLoader(datasets, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers),
                             pin_memory=True)

    net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(os.path.join(opt.model_dir, 'net_latest.pt'), map_location=map_location).items()})

    # net.load_state_dict(torch.load(os.path.join(opt.model_dir, 'net_latest.pt'), map_location=map_location))
    net.eval()
    time_test = 0
    count = 0
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
            HRI, LRI, HRsp, LRsp, Mask = [x.cuda() for x in data]
            TListX, TListZ, maskL = net(HRI, HRsp, LRI, LRsp, epoch=0)
            test_X = TListX[-1]
            test_Z = TListZ[-1]


            Xt = tensor2np(test_X)
            HRt = tensor2np(HRI)

            test_loss_normal = criterion(test_X, HRI)
            test_err += test_loss_normal.item()
            test_batches += 1

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
    test_dataset = DataSet(opt.test_data_path, opt.mask_dir_tst, opt.patchSize, opt.batchSize*opt.batchnum)
    criterion = MyLoss()

    net = net(opt).cuda()

    test_model(net, test_dataset)
