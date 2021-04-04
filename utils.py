import torch
import torch.nn as nn
import os, sys
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
import copy
import time

__all__ = ['Dataset', 'VDCNN', 'make_data',
           'make_dataloader', 'run_model']

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(-1, self.num_flat_features(x))
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
            
        return num_features    

class Dataset(torch.utils.data.Dataset):
    """
    Characterizes a dataset for PyTorch.
    """
    def __init__(self, X, Y):
        self.X = X 
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X_i = self.X[index]
        Y_i = self.Y[index]

        return X_i, Y_i

class KMaxPooling(nn.Module):
    """
    K-Max Pooling used as the last layer for VDCNN model implementation
    """
    def __init__(self, k=8):
        super(KMaxPooling, self).__init__()
        self.k = k
    def forward(self, x):
        idxes = x.topk(self.k, dim=2)[1].sort(dim=0)[0]
        return x.gather(2, idxes)


class VDCNN(nn.Module):
    """
    Implmentation of VDCNN model by Conneau et al. 2017
    """
    def __init__(self, depth=9, num_class=10):
        super(VDCNN, self).__init__()
        self.depth = depth
        self.num_class = num_class
        
        self.embedding = nn.Embedding(71, 16)
        
        if depth == 9:
            num_blocks = [2, 2, 2, 2]
        elif depth == 17:
            num_blocks = [4, 4, 4, 4]
        elif depth == 29:
            num_blocks = [4, 4, 10, 10]
        else:
            print('Choose depth among 9, 17, or 29.')
            sys.exit()

        self.conv64_block = nn.ModuleList()
        self.conv64_block += [nn.Conv1d(16, 64, 3, padding=1)]
        for i in range(num_blocks[0]):
            self.conv64_block += [nn.Conv1d(64, 64, 3, padding=1), 
                                  nn.BatchNorm1d(64, affine=False),
                                  nn.ReLU()]

        self.pool_half = nn.MaxPool1d(3, stride=2, padding=1) 
        
        self.conv128_block = nn.ModuleList()
        for i in range(num_blocks[1]):
            if i == 0:
                self.conv128_block += [nn.Conv1d(64, 128, 3, padding=1)]
            else:
                self.conv128_block += [nn.Conv1d(128, 128, 3, padding=1)]
            self.conv128_block += [nn.BatchNorm1d(128, affine=False),
                                   nn.ReLU()]   
        
        self.conv256_block = nn.ModuleList()
        for i in range(num_blocks[2]):
            if i == 0:
                self.conv256_block += [nn.Conv1d(128, 256, 3, padding=1)]
            else:
                self.conv256_block += [nn.Conv1d(256, 256, 3, padding=1)]
            self.conv256_block += [nn.BatchNorm1d(256, affine=False),
                                   nn.ReLU()]
        
        self.conv512_block = nn.ModuleList()
        for i in range(num_blocks[3]):
            if i == 0:
                self.conv512_block += [nn.Conv1d(256, 512, 3, padding=1)]
            else:
                self.conv512_block += [nn.Conv1d(512, 512, 3, padding=1)]
            self.conv512_block += [nn.BatchNorm1d(512, affine=False),
                                   nn.ReLU()]
        
        self.output_block = nn.ModuleList()
        
        self.output_block += [KMaxPooling(k=8),
                              Flatten(),
                              nn.Linear(4096, 2048),
                              nn.ReLU(),
                              nn.Linear(2048, 2048),
                              nn.ReLU(),
                              nn.Linear(2048, self.num_class),
                              nn.Softmax(dim=1)]
        
    def forward(self, x):

        x = self.embedding(x).view(-1, 16, 1024)
        for unit in self.conv64_block:
            x = unit(x)
        x = self.pool_half(x)
        for unit in self.conv128_block:
            x = unit(x)
        x = self.pool_half(x)
        for unit in self.conv256_block:
            x = unit(x)
        x = self.pool_half(x)
        for unit in self.conv512_block:
            x = unit(x)
        for unit in self.output_block:
            x = unit(x)
        return x
    
    
def make_data(train_fname, test_fname, use_oldfile=False):
    """
    Preprocess Yelp_review_polarity dataset and make them available for the model
    to train on. The output is numpy array
    """
    # make train and test data
    if (os.path.exists('train_X.npy') and os.path.exists('train_Y.npy') and
        os.path.exists('test_X.npy') and os.path.exists('test_Y.npy') and use_oldfile):
        train_X = np.load('train_X.npy')
        train_Y = np.load('train_Y.npy')
        test_X = np.load('test_X.npy')
        test_Y = np.load('test_Y.npy')
    else:
        train_df = pd.read_csv(train_fname, header=None)
        test_df = pd.read_csv(test_fname, header=None)
        lookup_table = {}
        chrs = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'"+'"/\|_@#$%^&*~`+-=<>()[]{} '
        for i, c in enumerate(chrs):
            lookup_table[c] = i+1 # reserve 0 as padding
        train_X = np.zeros([len(train_df), 1024]).astype(int)
        test_X = np.zeros([len(test_df), 1024]).astype(int)

        for i, s in enumerate(train_df.loc[:, 1]):
            for j, c in enumerate(s.lower()):
                train_X[i, j] = lookup_table[c]
                if j == 1023:
                    break
        for i, s in enumerate(test_df.loc[:, 1]):
            for j, c in enumerate(s.lower()):
                test_X[i, j] = lookup_table[c]
                if j == 1023:
                    break

        train_Y = train_df.loc[:, 0].to_numpy() - 1 
        test_Y = test_df.loc[:, 0].to_numpy() - 1 

        np.save('train_X', train_X)
        np.save('train_Y', train_Y)
        np.save('test_X', test_X)
        np.save('test_Y', test_Y)
    
    return train_X, train_Y, test_X, test_Y


def make_dataloader(train_fname, test_fname, num_workers=8, batch_size=128, use_oldfile=False):
    """
    Create a dataloader which is conveniently designed for pytorch batch 
    iteration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_X, train_Y, test_X, test_Y = make_data(train_fname, test_fname, use_oldfile)

    train_X = torch.from_numpy(train_X).to(device)    
    train_Y = torch.from_numpy(train_Y).to(device)
    test_X = torch.from_numpy(test_X).to(device)
    test_Y = torch.from_numpy(test_Y).to(device)

    dataset_train = Dataset(train_X, train_Y)
    dataset_test = Dataset(test_X, test_Y)
    
    dataloaders = {
        'train': DataLoader(dataset_train, batch_size, shuffle=False,
                            pin_memory=True,
                            num_workers=num_workers),
        'test': DataLoader(dataset_test, batch_size, shuffle=False,
                          pin_memory=True,
                          num_workers=num_workers),  
    }
    
    return dataloaders


def run_model(model, dataloaders, num_epochs):
    """
    Train and test the model with a given number of epochs. The result is saved onto
    3 log files.
    """
    epoch_file = open(f'epoch_d{model.depth}_nc{model.num_class}_ne{num_epochs}.log', 'w')
    short_file = open(f'short_d{model.depth}_nc{model.num_class}_ne{num_epochs}.log', 'w')
    # Record loss every minute 
    prev_time = time.time()
    mins = 0
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    loss_history = {x:np.zeros(num_epochs) for x in ['train','test']}

    for epoch in range(num_epochs):
        epoch_file.write('Epoch {}/{}   ---   \n'.format(epoch+1, num_epochs))

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            num_miss = 0
            cnt = 0
            samp = 0 
            # Iterate over data.
            for batch in dataloaders[phase]:
                inputs = batch[0]
                labels = batch[1]
                cnt += 1
                samp += inputs.size(0)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history only if in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if phase == 'train':
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()
                        # statistics
                        running_loss += loss.item()
                        avg_loss = running_loss/cnt
                    if phase == 'test':
                        yhat = torch.argmax(outputs, dim=1).detach().cpu().numpy().reshape(-1, 1)
                        labels = labels.detach().cpu().numpy().reshape(-1, 1)
                        num_miss += np.sum(abs(yhat-labels)).item()
                        err_perc = num_miss/samp*100
                         
                if time.time()-prev_time >= 60:
                    mins += 1
                    if phase == 'train':
                        short_file.write(f'[{phase}] {mins} minute(s): loss = {avg_loss:.5f}\n')
                    if phase == 'test':
                        short_file.write(f'[{phase}] {mins} minute(s): error % = {err_perc:.5f}\n')
                    prev_time = time.time()
                           
            if phase == 'train':
                epoch_loss = avg_loss
            if phase == 'test':
                epoch_loss = err_perc
            loss_history[phase][epoch] = epoch_loss

            epoch_file.write(f'{phase} Loss: {epoch_loss:.5f}\n')

    f.close(short_file)
    f.close(epoch_file)
    np.save('loss_history_train', loss_history[0])
    np.save('loss_history_test', loss_history[1])
