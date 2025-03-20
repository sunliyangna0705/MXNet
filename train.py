import argparse
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from math import ceil
from network.mxnet import net
from dataset.DataSet import DataSet
from utils.metric import complex_psnr
from utils.my_loss import MyLoss
from utils.visual import tensor2np
from fastmri.data import transforms as T
from datetime import datetime



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument("--train_data_path", type=str, default="/data/Admm-data/train", help='train data')
parser.add_argument("--validate_data_path", type=str, default="/data/Admm-data/train",
                    help='validate data')
parser.add_argument("--test_data_path", type=str, default="/data/Admm-data/test", help='test data')
parser.add_argument("--mask_dir_trn", type=str, default="/data/Admm-data/mask_40/mask0.4.mat",
                    help='train mask data')
parser.add_argument("--mask_dir_tst", type=str, default="/data/Admm-data/mask_40/mask0.4.mat",
                    help='test mask data')
parser.add_argument('--acc', type=float, default=0.4, help='initial learning rate')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--patchSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=1000, help='total number of training epochs')
parser.add_argument('--batch', type=int, default=100,
                    help='batchsize*batchnum=1000 for randomly selecting 1000 imag pairs at every iteration')
parser.add_argument('--num_channel', type=int, default=64,
                    help='the number of dual channels')
parser.add_argument('--num_mask_channel', type=int, default=1, help='the number of channel for mask')
parser.add_argument('--T', type=int, default=2, help='the number of ResBlocks in every ProxNet')
parser.add_argument('--S', type=int, default=10, help='the number of total iterative stages')
parser.add_argument('--resume', type=int, default=0, help='continue to train')
parser.add_argument("--milestone", type=int, default=[40, 80], help="When to decay learning rate")
parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate')
parser.add_argument('--log_dir', default='/models/DuSR-mask-learning/logs/', help='tensorboard logs')
parser.add_argument('--model_dir', default='/models/DuSR-mask-learning/models/', help='saving model')
parser.add_argument('--save_img_dir', default='/models/DuSR-mask-learning/imgs/',
                    help='saving temp images')
parser.add_argument('--fai', type=float, default=1, help='initialization fai')

opt = parser.parse_args()
def create_model_directory():
    try:
        os.makedirs(opt.model_dir)
    except OSError:
        pass

def normalize(image):
    img = (image - np.min(image)) / (np.max(image) - np.min(image))
    return img

def generate_session_filename():
    timestamp = datetime.now().strftime("%d%b_%I%M%S%P_")
    return os.path.join(opt.model_dir, f"{timestamp}{opt.acc}acc_{opt.niter}ep_{opt.S}T_{opt.S}S")


def initialize_data_loaders(train_dataset, validate_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True,
                              num_workers=int(opt.workers), pin_memory=True)
    validate_loader = DataLoader(validate_dataset, batch_size=opt.batchSize, shuffle=True,
                                 num_workers=int(opt.workers), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True,
                             num_workers=int(opt.workers), pin_memory=True)
    return train_loader, validate_loader, test_loader


def train_one_epoch(net, optimizer, scheduler, train_loader, writer, epoch, step):

    total_loss_org = 0
    train_batches = 0
    train_psnr = 0
    mse_per_epoch = 0
    tic = time.time()

    for ii, data in enumerate(train_loader):
        HRI, LRI, HRsp, LRsp, Mask = [x.cuda() for x in data]
        net.train()
        optimizer.zero_grad()

        X_out, Z_out, mask_out = net(LRsp)
        LZ1 = Z_out[-1]
        LX1 = X_out[-1]

        loss_img_all = torch.norm((LX1 - HRI), 'fro') / torch.norm(HRI, 'fro')
        loss_sp_all = torch.norm((LZ1 - HRsp), 'fro') / torch.norm(HRsp, 'fro')
        loss_mask = torch.norm((Mask[:, :, :, :, 0] - mask_out), 'fro') / torch.norm(Mask[:, :, :, :, 0], 'fro')
        # loss = loss_img_all + loss_sp_all + 0.1*loss_mask
        loss = loss_img_all + loss_sp_all

        loss.backward()
        optimizer.step()
        mse_iter = loss.item()
        mse_per_epoch += mse_iter
        train_batches += 1

        writer.add_scalar('Loss', loss, step)

        step += 1
        print(f"[epoch {epoch + 1}][{ii + 1}/{len(train_loader)}] loss: {loss:.4f} ")

        Xt = tensor2np(LX1)
        HRt = tensor2np(HRI)


        train_psnr_value = complex_psnr(np.abs(HRt), np.abs(Xt), peak='normalized')
        train_psnr += train_psnr_value

    train_psnr /= train_batches
    scheduler.step()
    writer.add_scalar('psnr on train data', train_psnr, epoch)
    print(f"train_psnr: {train_psnr}")
    toc = time.time()
    print(f'This epoch take time {toc - tic:.2f}')
    return train_psnr, step



def validate_model(net, validate_loader, criterion, writer, epoch):
    net.eval()
    test_err = 0
    test_psnr = 0
    test_batches = 0
    with torch.no_grad():
        for ii, data in enumerate(validate_loader):
            HRI, LRI, HRsp, LRsp, Mask = [x.cuda() for x in data]
            TListX, TListZ, maskL = net(LRsp)
            test_X = TListX[-1]
            test_Z = TListZ[-1]

            Xt = tensor2np(test_X)
            HRt = tensor2np(HRI)

            test_loss_normal = criterion(test_X, HRI)
            test_err += test_loss_normal.item()
            test_batches += 1
            test_psnr_value = complex_psnr(np.abs(HRt), np.abs(Xt), peak='normalized')
            test_psnr += test_psnr_value

    test_err /= test_batches
    test_psnr /= test_batches
    print(f"test_loss {test_err}")
    print(f"test_psnr {test_psnr}")
    writer.add_scalar('psnr on test data', test_psnr, epoch)
    return test_err, test_psnr



def train_model(net, optimizer, scheduler, train_dataset, validate_dataset, test_dataset):
    train_loader, validate_loader, test_loader = initialize_data_loaders(train_dataset, validate_dataset, test_dataset)
    num_data = len(train_dataset)
    num_iter_epoch = ceil(num_data / opt.batchSize)
    writer = SummaryWriter(opt.log_dir)
    step = 0
    PSNR = []

    for epoch in tqdm(range(opt.resume, opt.niter)):
        print(f"Epoch [{epoch} / {opt.niter}]")
        train_psnr, step = train_one_epoch(net, optimizer, scheduler, train_loader, writer, epoch, step)
        PSNR.append(train_psnr)


        torch.save(net.state_dict(), os.path.join(sessFileName, 'net_latest.pt'))
        if epoch % 100 == 0:
            model_prefix = 'model_'
            save_path_model = os.path.join(sessFileName, model_prefix + str(epoch + 1))
            torch.save({
                'epoch': epoch + 1,
                'step': step + 1,
            }, save_path_model)
            torch.save(net.state_dict(), os.path.join(sessFileName, f'net_{epoch + 1}.pt'))


        validate_model(net, validate_loader, criterion, writer, epoch)

    writer.close()
    print('Reach the maximal epochs! Finish training')
    print(f'The {np.argmax(PSNR)} model PSNR is Highest !')






if __name__ == '__main__':

    create_model_directory()
    sessFileName = generate_session_filename()
    if not os.path.exists(sessFileName):
        os.makedirs(sessFileName)


    net = net(opt).cuda()
    net = torch.nn.DataParallel(net)
    net = net.cuda()



    def print_network(name, net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(f'name={name}, Total number={num_params}')


    print_network("MXNet:", net)


    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.5)  # 学习率调度器

    for _ in range(opt.resume):
        scheduler.step()
    if opt.resume:
        try:
            net.load_state_dict(torch.load(os.path.join(opt.model_dir, f'net_{opt.resume + 1}.pt')))
            print(f'loaded checkpoints, epoch{opt.resume}')
        except FileNotFoundError:
            print(f"Checkpoint file net_{opt.resume + 1}.pt not found.")


    train_dataset = DataSet(opt.train_data_path, opt.mask_dir_trn, opt.patchSize, 100)
    validate_dataset = DataSet(opt.validate_data_path, opt.mask_dir_tst, opt.patchSize, 50)
    test_dataset = DataSet(opt.test_data_path, opt.mask_dir_tst, opt.patchSize, 50)

    criterion = MyLoss()
    train_model(net, optimizer, scheduler, train_dataset, validate_dataset, test_dataset)
