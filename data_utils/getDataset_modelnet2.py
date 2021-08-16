import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import json
import os
from PIL import Image
import numpy as np

class GetDataTrain(Dataset):
    
    def __init__(self, dataType='train', imageMode='RGB',views=20):

        self.dt = dataType
        self.imd = imageMode
        self.views = views 
        path = "/home/modelnet40/"

        clses = sorted(os.listdir(path))
        
        self.fls = []
        self.las = []
        self.names = []
        for c,cls in enumerate(clses):
            
            cls_path = os.path.join(path, cls, self.dt)
            objes = sorted(os.listdir(cls_path))

            for i in range(len(objes)//20):
                views_path = np.array([os.path.join(cls_path, v) for v in objes[i*20 :i*20+4]])
                self.fls.append(views_path)
                self.las.append(c)
                self.names.append(views_path[0])
        self.fls = np.array(self.fls)
        self.las = np.array(self.las)
    
    def trans(self, path):
        img = Image.open(path).convert('RGB')
        tf = transforms.Compose([
                transforms.RandomRotation(degrees=8), # fill=234),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        img = tf(img)
        return img
  

    def __getitem__(self, index):
        fl = self.fls[index][:self.views]
        target = torch.LongTensor([self.las[index]])
        imgs = []
        for p in fl:
            imgs.append(self.trans(p))
        data = torch.stack(imgs) 

        return {'data':data, 'target':target, 'name': self.names[index] }
                # data, big class , fine class,  domain

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.las)



if __name__ == '__main__':
    a = GetDataTrain(dataType='train', imageMode='RGB')
    for i,d in enumerate(a):
        print('i:%d / %d'%(i,len(a)),d['data'].shape, d['target'], d['name'])
