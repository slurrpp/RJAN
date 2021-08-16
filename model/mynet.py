import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.rvs import rvs
import pdb
import model.resnet as resnet
from torch.autograd import Variable

from torch import Tensor

class RJAN(nn.Module):
    def __init__(self, args, num_classes=40, pretrained=True):
        super(RJAN, self).__init__()
        self.pretrained = pretrained
        self.args = args
        self.Bk5 = Block9(args, num_classes)
       
        self.rra2 = RGA_Module(128, 28 * 28, use_spatial=True, use_channel=False,
                                    cha_ratio=4, spa_ratio=4, down_ratio=4)
        self.rra3 = RGA_Module(256, 14 * 14, use_spatial=True, use_channel=False,
                                    cha_ratio=2, spa_ratio=2, down_ratio=2)
        self.rra4 = RGA_Module(512, 7 * 7, use_spatial=True, use_channel=False,
                               cha_ratio=2, spa_ratio=2, down_ratio=2)
       
        self.cra2 = RGA_Module(128, 28 * 28, use_spatial=False, use_channel=True,
                                    cha_ratio=4, spa_ratio=4, down_ratio=4)
        self.cra3 = RGA_Module(256, 14 * 14, use_spatial=False, use_channel=True,
                                    cha_ratio=2, spa_ratio=2, down_ratio=2)
        self.cra4 = RGA_Module(512, 7 * 7, use_spatial=False, use_channel=True,
                               cha_ratio=2, spa_ratio=2, down_ratio=2)
        # -------------------------------------------------------------------------      
        self.av1pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1792, 512)
     
    def forward(self, x):
        x = x.reshape(-1, 3, 224, 224)
        x = self.Bk1(x)
        x = self.Bk2(x)

        x1_1 = self.rra2(x)
        x1 = self.Bk3(x1_1)
        x1_2 = self.rra3(x1)
        x1 = self.Bk4(x1_2)
        x1_3 = self.rra4(x1)
        x2_1 = self.cra2(x)
        x2 = self.Bk3(x2_1)
        x2_2 = self.cra3(x2)
        x2 = self.Bk4(x2_2)
        x2_3 = self.cra4(x2)

        x1_1 = self.av1pool(x1_1).reshape(-1, self.args.views, 128)
        x1_2 = self.av1pool(x1_2).reshape(-1, self.args.views, 256)
        x1_3 = self.av1pool(x1_3).reshape(-1, self.args.views, 512)
        x2_1 = self.av1pool(x2_1).reshape(-1, self.args.views, 128)
        x2_2 = self.av1pool(x2_2).reshape(-1, self.args.views, 256)
        x2_3 = self.av1pool(x2_3).reshape(-1, self.args.views, 512)
        
        x = torch.cat((x1_1, x1_2, x1_3, x2_1, x2_2, x2_3), dim=2)
        x = self.fc(x)
        x, fts = self.Bk5(x)
        return x, fts

class RGA_Module(nn.Module):
    def __init__(self, in_channel, in_spatial, use_spatial=True, use_channel=True, \
                 cha_ratio=8, spa_ratio=8, down_ratio=8):
        super(RGA_Module, self).__init__()

        self.in_channel = in_channel
        self.in_spatial = in_spatial

        self.use_spatial = use_spatial
        self.use_channel = use_channel

        print('Use_Spatial_Att: {};\tUse_Channel_Att: {}.'.format(self.use_spatial, self.use_channel))

        self.inter_channel = in_channel // cha_ratio
        self.inter_spatial = in_spatial // spa_ratio

        # Embedding functions for original features
        if self.use_spatial:
            self.gx_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.gx_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

        # Embedding functions for relation features
        if self.use_spatial:
            self.gg_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial * 2, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
        if self.use_channel:
            self.gg_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel * 2, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )

        # Networks for learning attention weights
        if self.use_spatial:
            num_channel_s = 1 + self.inter_spatial
            self.W_spatial = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s // down_ratio,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_channel_s // down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_s // down_ratio, out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )
        if self.use_channel:
            num_channel_c = 1 + self.inter_channel
            self.W_channel = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_c, out_channels=num_channel_c // down_ratio,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_channel_c // down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_c // down_ratio, out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )

        # Embedding functions for modeling relations
        if self.use_spatial:
            self.theta_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
            self.phi_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
        if self.use_channel:
            self.theta_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
            self.phi_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )

    def forward(self, x):
        b, c, h, w = x.size()

        if self.use_spatial:
            # spatial attention
            theta_xs = self.theta_spatial(x)
            phi_xs = self.phi_spatial(x)
            theta_xs = theta_xs.view(b, self.inter_channel, -1)
            theta_xs = theta_xs.permute(0, 2, 1)
            phi_xs = phi_xs.view(b, self.inter_channel, -1)
            Gs = torch.matmul(theta_xs, phi_xs)
            Gs_in = Gs.permute(0, 2, 1).view(b, h * w, h, w)
            Gs_out = Gs.view(b, h * w, h, w)
            Gs_joint = torch.cat((Gs_in, Gs_out), 1)
            Gs_joint = self.gg_spatial(Gs_joint)

            g_xs = self.gx_spatial(x)
            g_xs = torch.mean(g_xs, dim=1, keepdim=True)
            ys = torch.cat((g_xs, Gs_joint), 1)

            W_ys = self.W_spatial(ys)
            if not self.use_channel:
                out = F.sigmoid(W_ys.expand_as(x)) * x
                return out
            else:
                x = F.sigmoid(W_ys.expand_as(x)) * x

        if self.use_channel:
            # channel attention
            xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)
            theta_xc = self.theta_channel(xc).squeeze(-1).permute(0, 2, 1)
            phi_xc = self.phi_channel(xc).squeeze(-1)
            Gc = torch.matmul(theta_xc, phi_xc)
            Gc_in = Gc.permute(0, 2, 1).unsqueeze(-1)
            Gc_out = Gc.unsqueeze(-1)
            Gc_joint = torch.cat((Gc_in, Gc_out), 1)
            Gc_joint = self.gg_channel(Gc_joint)

            g_xc = self.gx_channel(xc)
            g_xc = torch.mean(g_xc, dim=1, keepdim=True)
            yc = torch.cat((g_xc, Gc_joint), 1)

            W_yc = self.W_channel(yc).transpose(1, 2)
            out = F.sigmoid(W_yc) * x

            return out


class Block9(nn.Module):
    def __init__(self, args, num_classes=40):
        super(Block9, self).__init__()

        self.args = args
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes),
        )
        self.fc1 = nn.Linear(1024, 512)
        self.emb1 = nn.Linear(512, 256)
        self.emb2 = nn.Linear(512, 256)
        self.bn = nn.BatchNorm2d(self.args.views)

    def forward(self, x):
        x = x.reshape(-1, self.args.views, 512)
        x1 = self.emb1(x)
        x2 = self.emb2(x)
        f = torch.matmul(x1, x2.permute(0, 2, 1))
        i = torch.sum(f, dim=1)
        i = i.reshape(-1, self.args.views, 1, 1)
        i = self.bn(i).reshape(-1, self.args.views)
        i = F.softmax(i, dim=1)
        i = i.reshape(-1, self.args.views, 1).repeat(1, 1, 512)
        x = torch.cat((x * i, x), dim=-1)
        x, _ = torch.max(x, dim=1)
        x = self.fc1(x)
        fts = x
        x = self.classifier(fts)
        return x, fts

