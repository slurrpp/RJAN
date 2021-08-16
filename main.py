from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
import argparse
import time

from losses import SupConLoss

from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils.getDataset_modelnet2 import GetDataTrain

from model.mynet import RJAN
from model.triCenter_loss import TripletCenterLoss


from scipy.io import savemat
import itertools


'''

'''


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batchsize', type=int, default=12, help='batch size in training')
    parser.add_argument('--epoch', default=60, type=int, help='number of epoch in training')
    parser.add_argument('--j', default=4, type=int, help='number of epoch in training')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training SGD or Adam')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--stage', type=str, default='train', help='train test, extract feature')
    parser.add_argument('--views', default=4, type=int, help='the number of views')
    parser.add_argument('--num_classes', default=40, type=int, help='the number of clsses')
    parser.add_argument('--model_name', type=str, default='pixelShuffle_rotation', help='train test')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--c', default=5.0, type=float)
    parser.add_argument('--r', default=4, type=int)
    parser.add_argument('--word_dim', default=512, type=int)
    return parser.parse_args()


args = parse_args()
args.device = torch.device('cuda:%s' % args.gpu)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    global args
    logger_train = get_logger('%s_train' % (args.model_name))
    logger_test = get_logger('%s_test' % (args.model_name))

    top_acc = 0.0
    top_acc_path = ''
    acc_avg = AverageMeter()
    losses = ['loss_o', 'loss_o1', 'loss_o2']

    trainDataset = GetDataTrain(dataType='train')
    validateDataset = GetDataTrain(dataType='test')
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=args.batchsize, shuffle=True, num_workers=args.j,
                                              pin_memory=True, drop_last=True)
    validateLoader = torch.utils.data.DataLoader(validateDataset, batch_size=args.batchsize, shuffle=False,
                                                 num_workers=args.j, pin_memory=True, drop_last=False)

    model = RJAN(args=args)

    if args.gpu == '0,1':
        device_ids = [int(x) for x in args.gpu.split(',')]
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    elif args.gpu == '0' or args.gpu == '1':
        model.to(args.device)

    optimizer = []
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer.append(torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.wd
        ))

    # train
    for epoch in range(0, args.epoch):
       
        ftsa = []
        laa = []
        acc_avg.reset()

        for idx, input_data in enumerate(tqdm(trainLoader)):
            data = input_data['data'].to(args.device)
            target = input_data['target'].reshape(-1)
            target = target.to(args.device)
            model.train()
            out, fts = model(data)
            loss_o = F.cross_entropy(out, target)
            loss = loss_o
            loss_o1 = 0
            loss_o2 = 0

            for op in optimizer:
                op.zero_grad()
            loss.backward()
            for op in optimizer:
                op.step()

            acc = get_acc_topk(out.cpu().data, target.cpu().data)
            acc_avg.update(acc)

            if (idx + 1) % 100 == 0:
                print_loss = 'epoch:%d, loess:%.4f' % (epoch, loss)
                for i in losses:
                    print_loss += ', %s : %.4f' % (i, eval(i))
                print(print_loss)

                logger_train.info(print_loss)

        if (epoch + 1) % 1 == 0:
            loss, acc, acc1, acc2 = Validate(args, model.eval(), validateLoader)
            print('loss', loss, acc, acc1, acc2)
            logger_test.info('---save model epoch:%d, acc:%.5f ' % (epoch, acc[-1]))
            if acc[-1] > top_acc:
                top_acc = acc[-1]
                print('save model...')
                if top_acc_path != '':  
                    os.remove(top_acc_path)
                top_acc_path = save_model(model, args.model_name, epoch, top_acc, top=True)
        if (epoch + 1) == args.epoch:
            top_acc_path = save_model(model, args.model_name, epoch, acc[-1], top=False)


def save_model(model, model_name, epoch, acc, top=True):
    checkpoints = 'experiment/checkpoints'
    print('Save model epoch:%d, acc:%.3f ... ' % (epoch, acc))
    fs = os.path.join(checkpoints, '%s_epoch_%d_acc_%.4f.pth' % (model_name, epoch, acc))
    torch.save(model.state_dict(), fs)
    if top:
        torch.save(model.state_dict(), 'experiment/checkpoints/%s_top.pth' % model_name)
        print('Save model of top acc ...')
    return fs


def load_model(model, path):
    pretrained = torch.load(path)
    model.load_state_dict(pretrained)


def get_acc_of_out(out, target):
    choice = out.max(1)[1]
    correct = choice.eq(target.long()).sum()
    return correct.item() / float(len(target))


def get_acc_topk(out, target, topk=(1,)):
    batch_size = target.shape[0]
    topkm = max(topk)
    _, pred = out.topk(topkm, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    acc = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        acc.append(correct_k.mul_(1.0 / batch_size))
    return np.array(acc)


def Validate(args, model, validateLoader):
    acc_avg = AverageMeter()
    acc1_avg = AverageMeter()
    acc2_avg = AverageMeter()
    loss_avg = AverageMeter()
    for idx, intput_data in enumerate(tqdm(validateLoader)):
        data = intput_data['data'].to(args.device)
        target = intput_data['target'].reshape(-1)  # .to(args.device)
        data = data.flatten(0, 1)
        with torch.no_grad():
            out, _ = model(data)
        out = out.cpu().data
        print(target)
        acc = get_acc_topk(out, target, (1,))
        acc_avg.update(np.array([acc]).reshape(-1))
    return 0, acc_avg.avg, 0, 0

if __name__ == '__main__':
    main()


