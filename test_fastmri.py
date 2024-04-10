import os
import os.path
import argparse
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import h5py
from scipy.io import loadmat
import PIL
from PIL import Image
from network.dusrnet import DuSRNet
from network.softnet import softnet
from network.ablation import ablationnet
from torch.utils.data import DataLoader
# from deeplesion.build_gemotry import initialization, build_gemotry
from fastmri.data import transforms as T
from dataset.DataSet import DuDataset
# from dataset.DataSet import CropDataset
from utils.my_loss import MyLoss
from utils.metric import complex_psnr
from utils.metric import complex_ssim
from utils.metric import nrmse
import MyLib as ML
import scipy.io as scio

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description="YU_Test")
# parser.add_argument("--model_dir", type=str, default="model", help='path to model and log files')
parser.add_argument("--model_dir", type=str, default="/data/sunlina/models/DuSR-mask-learning-Node11/models/26Oct_040019pm_0.4acc_1000ep_10T_10S",
                    help='path to model and log files')
parser.add_argument("--data_path", type=str, default="/data/sunlina/data/Admm-data/test", help='path to testing data')
parser.add_argument("--mask_dir_tst", type=str, default="/data/sunlina/data/InitMask/LearningMask40_tst.npz", help='test mask data')
parser.add_argument("--write_file", type=str, default="OutFile.h5", help='write file for h5')
parser.add_argument("--use_GPU", type=bool, default=False, help='use GPU or not')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--save_img_dir', default='/data/sunlina/models/DuSR-mask-learning-Node11/test_imgs/', help='saving temp images')
parser.add_argument('--acc', type=float, default=0.4, help='initial learning rate')
# parser.add_argument("--save_path", type=str, default="test_result", help='path to save data')
parser.add_argument("--save_path", type=str, default="/data/sunlina/models/DuSR-mask-learning-Node11/test_result/imgs", help='path to save data')
parser.add_argument('--num_channel', type=int, default=64, help='the number of dual channels')
parser.add_argument('--num_mask_channel', type=int, default=1, help='the number of channel for mask')
parser.add_argument('--patchSize', type=int, default=255, help='the height / width of the input image to network')
parser.add_argument('--T', type=int, default=4, help='the number of ResBlocks in every ProxNet')
parser.add_argument('--S', type=int, default=10, help='the number of total iterative stages')
parser.add_argument('--fai', type=float, default=10, help='initialization for stepsize eta1')
parser.add_argument('--batchSize', type=int, default=1, help='batch size for test')
parser.add_argument('--batchnum', type=int, default=50,
                    help='batchsize*batchnum=1000 for randomly selecting 1000 imag pairs at every iteration')
parser.add_argument('--w', type=float, default=256, help='Weight of data')
parser.add_argument('--h', type=float, default=256, help='Height of data')
# parser.add_argument('--alpha', type=float, default=0.5, help='initialization for weight factor')
opt = parser.parse_args()

if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = 'cpu'


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")


# Y  = LRsp    Z = HRsp
# LI = LRI     X = HRI
input_dir = opt.save_path + '/LRsp/'
gt_dir = opt.save_path + '/LRI/'
outX_dir = opt.save_path + '/X/'
outZ_dir = opt.save_path + '/Z/'

mkdir(input_dir)
mkdir(gt_dir)
mkdir(outX_dir)
mkdir(outZ_dir)


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


def test_model(net, datasets):
    data_loader = DataLoader(datasets, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers),
                             pin_memory=True)
    net.load_state_dict(torch.load(os.path.join(opt.model_dir, 'net_latest.pt'), map_location=map_location))
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

    # for imag_idx in range(1):
    # print(imag_idx)
    write_file = h5py.File(os.path.join(opt.save_path, opt.write_file), 'w')

    for ii, data in enumerate(data_loader):
        print('saving images : [ %d / %d ]' % (ii, len(data_loader)))
        # gc.collect()
        with torch.no_grad():
            HRI, LRI, HRsp, LRsp, Mask = [x.cuda() for x in data]
            TListX, TListZ, mask = net(HRI, HRsp, LRI,LRsp ,epoch=1)
            test_X = TListX[-1]
            test_Z = TListZ[-1]

            MaskLearned = mask[0, 0, :, :].cpu().detach().numpy()
            ML.imwrite(MaskLearned, os.path.join(opt.save_img_dir, 'MaskLearned_%d.png' % ii))
            # X = test_X[0,0,:,:,:]
            X1 = test_X[0, 0, :, :].cpu()
            Xt = (T.tensor_to_complex_np(X1))

            HR1 = HRI[0, 0, :, :].cpu()
            HRt = (T.tensor_to_complex_np(HR1))

            test_loss_normal = criterion(test_X, HRI)
            test_err += test_loss_normal.item()
            test_batches += 1
            if ii % 10 == 0:
                LR1 = LRI[0, 0, :, :].cpu()
                LRt = (T.tensor_to_complex_np(LR1))
                print("----------------saving figure--------%d--------" % ii)
                dataFile = 'E://data.mat'


                saveXt = abs(Xt)
                saveHRt = abs(HRt)
                saveLRt = abs(LRt)
                data = saveXt-saveHRt
                scio.savemat(os.path.join(opt.save_path, 'Res_%d.mat' % (ii)), {'Res': data})
                ML.imwrite(saveXt, os.path.join(opt.save_path, 'Rec_%d.png' % (ii)))
                ML.imwrite(saveHRt, os.path.join(opt.save_path, 'HR_%d.png' % (ii)))
                ML.imwrite(saveLRt, os.path.join(opt.save_path, 'LR_%d.png' % (ii)))

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

        write_file['Out_X%d' % ii] = test_X.cpu()
        write_file['Out_Z%d' % ii] = test_Z.cpu()
    write_file.close()
    test_err /= test_batches
    test_psnr /= test_batches
    test_ssim /= test_batches
    test_nrmse /= test_batches

    print("test_loss ", test_err)
    print("test_psnr ", test_psnr, "mean: ", np.mean(PSNR), "var: ", np.std(PSNR))
    print("test_ssim ", test_ssim, "mean: ", np.mean(SSIM), "var: ", np.std(SSIM))
    print("test_nrmse ", test_nrmse, "mean: ", np.mean(NRMSE), "var: ", np.std(NRMSE))

    print("Successful Save file")
    # %% Display the output images
    plot = lambda x: plt.imshow(x, cmap=plt.cm.gray, clim=(0.0, 1.0))
    plt.clf()
    plt.subplot(141)
    plot(MaskLearned)
    plt.axis('off')
    plt.title('Mask')
    plt.subplot(142)
    plot(saveHRt)
    plt.axis('off')
    plt.title('Original')
    plt.subplot(143)
    plot(saveLRt)
    plt.title('Input')
    plt.axis('off')
    plt.subplot(144)
    plot(saveXt)
    plt.title('Output')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=.01)
    plt.show()


if __name__ == "__main__":
    # main()
    # load dataset
    # filename = '/data/sunlina/data/Admm-data/our_data_learning_mask40_tst20220609.npz'
    # f = np.load(filename)
    # org, csm, mask = f['org'][:], f['csm'][:], f['mask'][:]
    filename = '/data/sunlina/data/Admm-data/mask_40/mask0.4.mat'
    f = loadmat(filename)
    mask = f['mask']
    test_dataset = DuDataset(opt.data_path, mask,opt.patchSize, 1)
    criterion = MyLoss()
    net = softnet(opt).cuda()
    #net = ablationnet(opt).cuda()
    test_model(net, test_dataset)
