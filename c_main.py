from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
import time

from torch.utils.data import DataLoader
from tqdm import tqdm

# from data_utils.getDataset import GetDataTrain
from data_utils.getDataset_modelnet2 import GetDataTrain

from model.myresnet import myresnet
#from model.triCenter_loss import TripletCenterLoss
#from model.center_loss import CenterLoss
#from model.cosCenter_loss import CenterLoss as CosLoss

from scipy.io import savemat
import itertools

'''

'''

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batchsize', type=int, default=16, help='batch size in training')
    parser.add_argument('--epoch',  default=80, type=int, help='number of epoch in training')
    parser.add_argument('--j',  default=4, type=int, help='number of epoch in training')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training SGD or Adam')
    parser.add_argument('--pretrained', dest='pretrained', action ='store_true', help='use pre-trained model')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--wd', default=1e-4, type=float,metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--phase', type=str, default='train', help='train test')
    parser.add_argument('--views',  default=9, type=int, help='the number of views')
    parser.add_argument('--num_classes',  default=40, type=int, help='the number of clsses')
    return parser.parse_args()


args = parse_args()
args.device = torch.device('cuda:%s'%args.gpu)


ins6 = torch.LongTensor([[0., 1., 2., 3., 4., 5.],
        [1., 2., 3., 4., 5., 0.],
        [2., 3., 4., 5., 0., 1.],
        [3., 4., 5., 0., 1., 2.],
        [4., 5., 0., 1., 2., 3.],
        [5., 0., 1., 2., 3., 4.]])

ins12 = torch.LongTensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.],
        [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.,  0.],
        [ 2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.,  0.,  1.],
        [ 3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.,  0.,  1.,  2.],
        [ 4.,  5.,  6.,  7.,  8.,  9., 10., 11.,  0.,  1.,  2.,  3.],
        [ 5.,  6.,  7.,  8.,  9., 10., 11.,  0.,  1.,  2.,  3.,  4.],
        [ 6.,  7.,  8.,  9., 10., 11.,  0.,  1.,  2.,  3.,  4.,  5.],
        [ 7.,  8.,  9., 10., 11.,  0.,  1.,  2.,  3.,  4.,  5.,  6.],
        [ 8.,  9., 10., 11.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
        [ 9., 10., 11.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.],
        [10., 11.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
        [11.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]])


# 可以针对全局，也可以针对局部
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

# use: 输入optimizer 和 epoch 就可以使用
#
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 200 epochs"""
    lr = args.lr * (0.1 ** (epoch*3 //args.epoch))   # 每两百个
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print ('Learning Rate: {lr:.6f}'.format(lr=param_group['lr']))
    return lr

def dist_elur(fts_q, fts_c):
    fts_q = fts_q/torch.norm(fts_q, dim=-1,keepdim=True)
    fts_c = fts_c/torch.norm(fts_c, dim=-1,keepdim=True)
    fts_qs = torch.sum((fts_q)**2,dim=-1,keepdim=True)
    fts_cs = torch.sum((fts_c)**2,dim=-1,keepdim=True).t()
    qc = torch.mm(fts_q,fts_c.t())
    dist = fts_qs + fts_cs - 2 * qc +1e-4
    return torch.sqrt(dist)

def dist_cos(fts_q, fts_c):
    up = torch.matmul(fts_q,fts_c.T)
    down1 = torch.sqrt(torch.sum((fts_q)**2,axis=-1,keepdims=True))
    down2  = torch.sqrt(torch.sum((fts_c)**2,axis=-1,keepdims=True).t())
    down = torch.mm(down1, down2)
    dist = up/(down+1e-4)
    return 1 - dist


def mi_dist(fts1,fts2,la1,la2,margin=1,mode=2):
    dist = dist_elur(fts1,fts2)
    index = (la1[:,mode].reshape(-1,1)==la2[:,mode].reshape(1,-1)).bool()
    ap = dist[index]
    lens = len(ap)
    an = torch.sort(dist[(1-index.long()).bool()])[0][:lens]
    if lens*2 > (dist.shape[0]**2):
        ap = ap.mean().unsqueeze(0)
        an = an.mean().unsqueeze(0)
    loss = nn.MarginRankingLoss(margin)(ap,an,torch.Tensor([-1]).to(device)    )
    return loss,ap.mean()


path_model = [
    'experiment/checkpoints/top.pth' # 0   训练最好的
]

path_mat =[os.path.join('metric', os.path.basename(i).split('pth')[0] + 'mat') for i in path_model]



def g_t(target):
    b = args.batchsize
    tx = torch.ones(b, args.views, args.views)
    tx = (tx * args.num_classes).long()
    
    tx[:, torch.arange(args.views), torch.arange(args.views)] = target.repeat(args.views, 1).t()

    return tx

def g_tt(index, target):
    b = args.batchsize
    tx = torch.ones(b, args.views, args.views)
    tx = (tx * args.num_classes).long()
    ins = ins6 if args.views==6 else ins12
    tx[:, torch.arange(args.views), ins[index]] = target.repeat(args.views,1).t()

    return tx

def g_ts(index, target):
    b = args.batchsize
    v = args.views
    tx = torch.ones(b, v, v)
    tx = (tx * args.num_classes).long()
    ins = ins6 if args.views==6 else ins12
    tx[torch.arange(b).repeat(v, 1).t(), torch.arange(v).repeat(b, 1), ins[index]] = target.repeat(v, 1).t()
    return tx


def main():
    global args
    # 数据记录
    model_name = 'rotation_fixed'
    logger_train = get_logger('%s_train'%(model_name))
    logger_test = get_logger('%s_test'%(model_name))
    top_acc = 0.0
    top_acc_path = ''
    acc_avg = AverageMeter()
    losses = ['loss_o', 'loss_o1', 'loss_o2']


    # domainMode 0,1  是image, render/ mask  + train/test
    #           2,3   是image, render, mask  + 全部数据
    #           4,5   是image /render,      + train/test
    trainDataset =  GetDataTrain(dataType='train', imageMode='RGB', views=args.views)
    validateDataset =  GetDataTrain(dataType='test', imageMode='RGB', views=args.views)
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size =args.batchsize,shuffle=True,num_workers=args.j, pin_memory = True, drop_last=True)
    validateLoader = torch.utils.data.DataLoader(validateDataset, batch_size =args.batchsize, shuffle=False, num_workers=args.j,pin_memory=True,  drop_last=False)
    
    

    model = myresnet(args=args)
    if args.pretrained:
        if os.path.isfile(path_model[0]):
            load_model(model, path_model[0])
            print('! Using pretrained model')
        else: print('? No pretrained model!')
    else: print('! Do not use pretrained model')
    # cri_triCenter_d = TripletCenterLoss(margin=5).to(device)
    #cri_triCenter_cat = TripletCenterLoss(margin=5,num_classes=7,feat_dim=512).to(args.device)
    #cri_triCenter_cent = TripletCenterLoss(margin=2,num_classes=5202,feat_dim=512).to(args.device)
    #cri_centerc = CenterLoss(num_classes=34, feat_dim=512).to(args.device)
    #cri_cos_cat = CosLoss(num_classes = 7, feat_dim=512).to(args.device)
    #cri_cos_cent = CosLoss(num_classes = 5202, feat_dim=512).to(args.device)
    #cri_ce =  torch.nn.CrossEntropyLoss()
    #cri_triMargin =  torch.nn.TripletMarginLoss(margin=1.0, p=2.0) 
    # process: gpu use
    if args.gpu == '0,1':
        device_ids = [int(x) for x in args.gpu.split(',')]
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    elif args.gpu == '0' or args.gpu == '1':
        #device = torch.device('cuda:'+args.gpu)
        #model.to(device)
        model.to(args.device)
        #cri_triCenter_d.to(args.device)
        #cri_triCenter_c.to(args.device)
        
        
        
        # process: optimizer
    if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
            #optimizer_triCenter_c = torch.optim.SGD(itertools.chain(cri_triCenter_cat.parameters(),cri_triCenter_cent.parameters()), lr=0.1)
    elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.wd
            )
            
            #optimizer_triCenter = torch.optim.SGD(itertools.chain(cri_triCenter_cat.parameters(),cri_triCenter_cent.parameters()), lr=0.1)
            #optimizer_center_c = torch.optim.SGD(cri_center_c.parameters(), lr=0.1)
            #optimizer_cos_c = torch.optim.SGD(cri_cos_c.parameters(), lr=0.1)
    if args.phase=='test':
        #ret_feat('val',model.eval(), path_index,validateLoader_image,validateLoader_render)
        ret_feat('test', model.eval(), path_index,testLoader_img,testLoader_rend)
        exit(-1)
                     
    # 因为中断所以要临时添加
   

    # train
    for epoch in range(0, args.epoch):
        cur_lr = adjust_learning_rate(optimizer,epoch)
        ftsa = []
        laa = []
        acc_avg.reset()
        
        for idx, input_data in enumerate(tqdm(trainLoader)):
            data = input_data['data'].to(args.device)
            target = input_data['target'].reshape(-1)
            tx1 = g_t(target).to(args.device).reshape(-1)
            tx2 = target.repeat(args.views, 1).t().to(args.device).reshape(-1)
            target = target.to(args.device)
            data = data.flatten(0,1)
            model.train()
            out,fts, out1, out2= model(data)
            
            # out = out.reshape(-1, args.num_classes + 1)
            loss_o = F.cross_entropy(out, target)
            out1 = out1.reshape(-1, args.views, args.num_classes+1).flatten(0, 1) 
            out2 = out2.reshape(-1, args.views, args.num_classes+1).flatten(0, 1) 
            loss_o1 = F.cross_entropy(out1, tx1)
            loss_o2 = F.cross_entropy(out2, tx2)
            loss = loss_o + loss_o1 + loss_o2 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
             
            acc = get_acc_topk(out.cpu().data, target.cpu().data)
            acc_avg.update(acc) 

            if (idx+1) % 100 ==0:
                print_loss = 'epoch:%d, loess:%.4f'%(epoch, loss)
                for i in losses:
                    print_loss += ', %s : %.4f'%(i, eval(i))
                print(print_loss)

                logger_train.info(print_loss)
          
        # 作用：测试并保存数据
        if (epoch+1) % 1 == 0:
            loss,acc, acc1, acc2 = Validate(args, model.eval(), validateLoader)
            print('loss',loss, acc, acc1, acc2)
            logger_test.info('---save model epoch:%d, acc:%.5f, acc1:%.5f, acc2:%.5f ' % (epoch, acc[-1], acc2[-1], acc2[-1]))
            if acc[-1] > top_acc:
                top_acc = acc[-1]
                print('save model...')
                if top_acc_path !='':    # 删去之前的包，这里可以不适用
                    os.remove(top_acc_path)
                top_acc_path = save_model(model, model_name, epoch, top_acc)


def save_model(model,model_name,epoch,acc):
    checkpoints = 'experiment/checkpoints'
    fs = os.path.join(checkpoints,'%s_epoch_%d_acc_%.4f.pth'%(model_name,epoch,acc))
    torch.save(model.state_dict(),fs)
    torch.save(model.state_dict(), 'top.pth')
    return fs


def load_model(model,path):
    pretrained = torch.load(path)
    model.load_state_dict(pretrained)


def get_acc_of_out(out,target):
    choice = out.max(1)[1]
    correct = choice.eq(target.long()).sum()
    return correct.item() / float(len(target))

def get_acc_topk(out,target,topk=(1,)):
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


def getFts(model, test_loader):
    
    fts = []
    las = []
    for batch_idx, (data, target,_) in enumerate(tqdm(test_loader)):
        data = data.to(args.device)
        pred,ft,_ = model(data)
        fts.append(ft.cpu().detach().numpy())
        las.append(target.numpy())

    fts = np.concatenate(fts,axis=0)
    fts = fts.reshape(-1,fts.shape[-1])
    las = np.concatenate(las, axis=0)
    #save_fig(fts,las)       
    return {'fts':fts,'las':las}

def extract_feat(name,model,path_index, loader_img,loader_rend):
    pretrained = torch.load(path_model[path_index])
    model.load_state_dict(pretrained)
    re = getFts(model, loader_img)
    savemat(path_mat[path_index]+name+'img', re)
    re = getFts(model, loader_rend)
    savemat(path_mat[path_index]+name+'rend', re)
    print('.mat saved')
    exit(-1)




def Validate(args, model, validateLoader):
    # 各类准确率
    acc_avg = AverageMeter()
    acc1_avg = AverageMeter()
    acc2_avg = AverageMeter()
    loss_avg = AverageMeter()
    
    for idx, intput_data in enumerate(tqdm(validateLoader)):
        data   = intput_data['data'].to(args.device)
        target = intput_data['target'].reshape(-1)  #.to(args.device)
        # tx1  = g_t(target).reshape(-1)
        # tx2  = target.repeat(args.views, 1).t().reshape(-1)
        data = data.flatten(0,1) 
        
        with torch.no_grad():
            out,_, out1, out2 = model(data)
        
        out = out.cpu().data
        out1 = out1.cpu().data
        out1 = out1.reshape(-1, args.views, args.views, args.num_classes+1)[:, torch.arange(args.views), torch.arange(args.views), :].sum(dim=1)
        out2 = out2.cpu().data.reshape(-1, args.views, args.num_classes+1).sum(dim=1)
        
        acc  = get_acc_topk(out, target, (1,))
        acc1 = get_acc_topk(out1, target, (1,))
        acc2 = get_acc_topk(out2, target, (1,))
        
        acc_avg.update(np.array([acc]).reshape(-1))
        acc1_avg.update(np.array([acc1]).reshape(-1))
        acc2_avg.update(np.array([acc2]).reshape(-1))

    return 0, acc_avg.avg, acc1_avg.avg, acc2_avg.avg

def test(model, test_loader):
    batch_correct =[]
    batch_loss =[]
    
   
    for batch_idx, (data, target) in enumerate(test_loader):
        pred,y = model(data.view(-1,784))
        loss = (pred, target)
        choice = pred.data.max(1)[1]
        correct = choice.eq(target.long()).sum()
        batch_correct.append(correct.item()/float(len(target)))
        batch_loss.append(loss.data.item())
    # print('test:',np.mean(batch_correct))
    return np.mean(batch_correct), np.mean(batch_loss)



if __name__ == '__main__':
    main()

    # img = GetDataTrain(dataType='train', imageMode='RGB', domainMode=0)

