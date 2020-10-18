#!coding=utf-8
import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt

from dataset import Dataset
from IPython import embed

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(21*2, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 5, bias=True),
        )

    def forward(self, X):
        X = self.classifier(X)
        return X


def train(net, train_iter, test_iter, loss, conf, device='cuda'):
    
    num_epochs = conf['num_epochs']
    optimizer = torch.optim.Adam(net.parameters(), lr=conf['adam_lr'], betas=conf['adam_b1_b2'])
    
    net.train()
        
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            
            X = X.float()

            optimizer.zero_grad()            
            y_h = net(X)
            
            l = loss(y_h, y)
            
            l.backward()
            optimizer.step()  
            
            train_l_sum += l.item()
            n += y.shape[0]

        print('epoch %d, loss %.4f'%(epoch + 1, train_l_sum / n))

if __name__=='__main__':
    
    root = 'e:/projects/gestures-app/dl-scripts'

    conf = {
        'src': 'e:/projects/gestures-app/server/db.json',
        'checkpoint': '%s/checkpoint'%root,
        
        'batch_size': 64,
        'device': 'cuda',

        'resume':       True,
        'num_epochs':   100,
 
        'adam_lr':      1e-3,
        'adam_b1_b2':   (0.5,0.999),    #for adam
    }

    dataset = Dataset(conf)

    iter_train = torch.utils.data.DataLoader(dataset, batch_size=conf['batch_size'], shuffle=True, num_workers=0)
    iter_test  = torch.utils.data.DataLoader(dataset, batch_size=conf['batch_size'], shuffle=False, num_workers=0)
    
    loss = torch.nn.CrossEntropyLoss()
    net = Net().to(conf['device'])
    
    if conf['resume']:
        print('[-] ... resuming from last checkpoint')
        buf = torch.load('%s/net_v0.pth'%conf['checkpoint'])
        net.load_state_dict(buf)
    
    train(net, iter_train, iter_test, loss, conf, device=conf['device'])
    
    print('[-] ... saving checkpoint')
    torch.save(net.state_dict(), '%s/net_v0.pth'%conf['checkpoint'])
   
    embed()

