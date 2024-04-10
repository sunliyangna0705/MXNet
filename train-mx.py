import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from math import ceil
from network.mxnet import mxnet
from network.mx2 import mxnet
from dataset.DataSet import DataSet0
from utils.metric import complex_psnr
from utils.my_loss import MyLoss
from fastmri.data import transforms as T
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="/data/sunlina/data/Admm-data/train", help='train data')
    parser.add_argument("--validate_data_path", type=str, default="/data/sunlina/data/Admm-data/validate", help='validate data')
    parser.add_argument("--test_data_path", type=str, default="/data/sunlina/data/Admm-data/test", help='test data')
    parser.add_argument("--mask_dir_trn", type=str, default="/data/sunlina/data/InitMask/LearningMask20_trn.npz", help='train mask data')
    parser.add_argument("--mask_dir_tst", type=str, default="/data/sunlina/data/InitMask/LearningMask20_tst.npz", help='test mask data')
    parser.add_argument('--acc', type=float, default=0.4, help='initial learning rate')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--patchSize', type=int, default=256, help='the height / width of the input image to network')
    parser.add_argument('--niter', type=int, default=1000, help='total number of training epochs')
    parser.add_argument('--batchnum', type=int, default=100, help='batchsize*batchnum=1000 for randomly selecting 1000 imag pairs at every iteration')
    parser.add_argument('--num_channel', type=int, default=64, help='the number of dual channels')  # refer to https://github.com/hongwang01/RCDNet for the channel concatenation strategy
    parser.add_argument('--num_mask_channel', type=int, default=1, help='the number of channel for mask')
    parser.add_argument('--T', type=int, default=2, help='the number of ResBlocks in every ProxNet')
    parser.add_argument('--S', type=int, default=10, help='the number of total iterative stages')
    parser.add_argument('--resume', type=int, default=0, help='continue to train')
    parser.add_argument("--milestone", nargs='+', type=int, default=[40, 80], help="When to decay learning rate")
    parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate')
    parser.add_argument('--log_dir', default='/data/sunlina/models/DuSR-mask-learning/logs/', help='tensorboard logs')
    parser.add_argument('--model_dir', default='/data/sunlina/models/DuSR-mask-learning/models/', help='saving model')
    parser.add_argument('--save_img_dir', default='/data/sunlina/models/DuSR-mask-learning/imgs/', help='saving temp images')
    parser.add_argument('--fai', type=float, default=1, help='initialization fai')
    parser.add_argument('--w', type=float, default=256, help='Weight of data')
    parser.add_argument('--h', type=float, default=256, help='Height of data')
    return parser.parse_args()

def train_model(net, optimizer, scheduler, traindatasets, validatedatasets, testdatasets):
    train_data_loader = DataLoader(traindatasets, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers),
                                   pin_memory=True)
    validate_data_loader = DataLoader(validatedatasets, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers),
                                   pin_memory=True)
    test_data_loader = DataLoader(testdatasets, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers),
                                  pin_memory=True)

    num_data = len(traindatasets)
    num_iter_epoch = ceil(num_data / opt.batchSize)
    writer = SummaryWriter(opt.log_dir)
    step = 0
    PSNR = []

    for epoch in tqdm(range(opt.resume, opt.niter)):
        print("Epoch [%d / %d]" % (epoch, (opt.niter)))
        total_loss_org = 0
        train_batches = 0
        train_psnr = 0
        mse_per_epoch = 0
        tic = time.time()
        # train stage
        lr = optimizer.param_groups[0]['lr']
        ###############################################################################
        phase = 'train'
        ###############################################################################

        for ii, data in (enumerate(train_data_loader)):
            HRI, LRI, HRsp, LRsp, Mask = [x.cuda() for x in data]

            net.train()
            optimizer.zero_grad()
            ListX, ListZ, maskL = net(LRsp)
            LZ1 = ListZ[-1]
            LX1 = ListX[-1]

            loss_img_all = torch.norm((LX1 - HRI), 'fro') / torch.norm(HRI, 'fro')
            loss_sp_all = torch.norm((LZ1 - HRsp), 'fro') / torch.norm(HRsp, 'fro')

            loss = loss_img_all + loss_sp_all

            loss.backward()
            optimizer.step()
            mse_iter = loss.item()
            mse_per_epoch += mse_iter
            train_batches += 1
            writer.add_scalar('Loss', loss, step)
            step += 1
            X1 = LX1[0, 0, :, :].cpu()
            Xt = (T.tensor_to_complex_np(X1.detach()))

            HR1 = HRI[0, 0, :, :].cpu()
            HRt = (T.tensor_to_complex_np(HR1.detach()))
            train_psnr_value = complex_psnr(abs(HRt), abs(Xt),
                                            peak='normalized')
            train_psnr += train_psnr_value
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f " % (epoch + 1, ii + 1, len(train_data_loader), loss,train_psnr / (ii + 1)))
        train_psnr /= train_batches
        PSNR.append(train_psnr)
        scheduler.step()
        torch.save(net.state_dict(), os.path.join(sessFileName, 'net_latest.pt'))
        if (epoch % 20 == 0):
            # save model
            model_prefix = 'model_'
            save_path_model = os.path.join(sessFileName, model_prefix + str(epoch + 1))
            torch.save({
                'epoch': epoch + 1,
                'step': step + 1,
            }, save_path_model)
            torch.save(net.state_dict(), os.path.join(sessFileName, 'net_%d.pt' % (epoch + 1)))
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc - tic))
        ###############################################################################
        phase = 'validate'
        ###############################################################################
        valid_err = 0
        valid_psnr = 0
        valid_batches = 0
        net.eval()
        for ii, data in enumerate(validate_data_loader):
            with torch.no_grad():
                HRI, LRI, HRsp, LRsp, Mask = [x.cuda() for x in data]
                VListX, VListZ, maskL = net(LRsp)
                valid_X = VListX[-1]

                X1 = valid_X[0, 0, :, :].cpu()
                Xt = (T.tensor_to_complex_np(X1))

                HR1 = HRI[0, 0, :, :].cpu()
                HRt = (T.tensor_to_complex_np(HR1))

                valid_loss_normal = criterion(valid_X, HRI)
                valid_err += valid_loss_normal.item()
                valid_batches += 1
                valid_psnr_value = complex_psnr(abs(HRt), abs(Xt),
                                               peak='normalized')
                valid_psnr += valid_psnr_value
        valid_err /= valid_batches
        valid_psnr /= valid_batches
        print("test_loss ", valid_err)
        print("test_psnr ", valid_psnr)
        writer.add_scalar('psnr on test data', valid_psnr, epoch)

        ###############################################################################
        phase = 'test'
        ###############################################################################
        test_err = 0
        test_psnr = 0
        test_batches = 0
        net.eval()
        for ii, data in enumerate(test_data_loader):
            # gc.collect()
            with torch.no_grad():
                HRI, LRI, HRsp, LRsp, Mask = [x.cuda() for x in data]

                TListX, TListZ, maskL = net(LRsp)
                test_X = TListX[-1]
                test_Z = TListZ[-1]

                X1 = test_X[0, 0, :, :].cpu()
                Xt = (T.tensor_to_complex_np(X1))

                HR1 = HRI[0, 0, :, :].cpu()
                HRt = (T.tensor_to_complex_np(HR1))

                test_loss_normal = criterion(test_X, HRI)
                test_err += test_loss_normal.item()
                test_batches += 1
                test_psnr_value = complex_psnr(abs(HRt), abs(Xt),
                                               peak='normalized')
                test_psnr += test_psnr_value
        test_err /= test_batches
        test_psnr /= test_batches
        print("test_loss ", test_err)
        print("test_psnr ", test_psnr)
        writer.add_scalar('psnr on test data', test_psnr, epoch)

    writer.close()



# 定义规范化方法
def normalize(image):
    img = (image - np.min(image)) / (np.max(image) - np.min(image))
    return img


if __name__ == '__main__':
    opt = parse_args()

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)
    # create path
    try:
        os.makedirs(opt.model_dir)
    except OSError:
        pass

    cudnn.benchmark = True  # 增加程序运行效率，自动寻找合适的当前配置的高效算法

    directory = opt.model_dir + datetime.now().strftime("%d%b_%I%M%S%P_") + \
                str(opt.acc) + 'acc_' + str(opt.niter) + 'ep_' + str(opt.S) + 'T_' + str(opt.S) + 'S'

    if not os.path.exists(directory):
        os.makedirs(directory)
    sessFileName = directory

    net = mxnet(opt).cuda()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    criterion = MyLoss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.5)

    # load Dataset
    filename = '/data/sunlina/data/Admm-data/our_data_learning_mask80_trn20231218.npz'
    f = np.load(filename)
    org, csm, mask = f['org'][:], f['csm'][:], f['mask'][:]
    print("mask ", mask.shape)
    train_dataset = DataSet0(org, mask,opt.patchSize, 100)
    validate_dataset = DataSet0(org, mask, opt.patchSize, 100)
    test_dataset = DataSet0(org, mask,opt.patchSize, 50)

    # training model
    train_model(net, optimizer, scheduler, train_dataset, validate_dataset,test_dataset)

