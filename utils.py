import torch
import torch.nn as nn
import os, sys
import numpy as np
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
                              ]
        
    def forward(self, x):

        x = self.embedding(x).transpose(1, 2)
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

    def init_weights(self, m):
        """Use He initialization for conv and linear layers that have ReLU rectfier
           afterward.
        """
        if type(m) == nn.Linear:
            stdv = np.sqrt(2./m.weight.size(1))
            # sqrt(3) scaling to account for uniform distribution variance 
            m.weight.data.uniform_(-np.sqrt(3)*stdv,
                                   np.sqrt(3)*stdv)
            if m.bias is not None:
                m.bias.data.fill_(0)

        elif type(m) == nn.Conv1d:
            n = m.in_channels
            for k in m.kernel_size:
                n *= k
            stdv = np.sqrt(2./n)
            m.weight.data.uniform_(-np.sqrt(3)*stdv,
                                    np.sqrt(3)*stdv)
            if m.bias is not None:
                m.bias.data.fill_(0)
    
    
def make_data(train_fname, test_fname):
    """
    Preprocess Yelp_review_polarity dataset and make them available for the model
    to train on. The output is numpy array
    """
    dataset_name = train_fname.split('/')[3]

    # make train and test data
    train_df = pd.read_csv(train_fname, header=None)
    test_df = pd.read_csv(test_fname, header=None)
    lookup_table = {}
    chrs = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'"+'"/\|_@#$%^&*~`+-=<>()[]{} '
    for i, c in enumerate(chrs):
        lookup_table[c] = i+1 # reserve 0 as padding
    train_X = np.zeros([len(train_df), 1024]).astype(int)
    test_X = np.zeros([len(test_df), 1024]).astype(int)

    label_col = 0
    if dataset_name == 'yelp_review_polarity_csv':
        data_col = 1
    elif dataset_name == 'yahoo_answers_csv':
        data_col = 3 

    for i, s in enumerate(train_df.loc[:, data_col]):
        for j, c in enumerate(str(s).lower()):
            if c not in lookup_table.keys():
                train_X[i, j] = 0
            else:
                train_X[i, j] = lookup_table[c]
            if j == 1023:
                break
    for i, s in enumerate(test_df.loc[:, data_col]):
        for j, c in enumerate(str(s).lower()):
            if c not in lookup_table.keys():
                test_X[i, j] = 0
            else:
                test_X[i, j] = lookup_table[c]
            if j == 1023:
                break

    train_Y = np.array(train_df.loc[:, label_col]) - 1 
    test_Y = np.array(test_df.loc[:, label_col]) - 1 

    
    return train_X, train_Y, test_X, test_Y


def make_dataloader(train_fname, test_fname, batch_size=128):
    """
    Create a dataloader which is conveniently designed for pytorch batch 
    iteration.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_X, train_Y, test_X, test_Y = make_data(train_fname, test_fname)

    dataloaders = {
        'train': (train_X, train_Y, batch_size),
        'test': (test_X, test_Y, batch_size),  
    }
    
    return dataloaders


def run_model(model, dataloaders, num_epochs):
    """
    Train and test the model with a given number of epochs. The result is saved onto
    3 log files.
    """
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # apply He initialization
    model.apply(model.init_weights)

    epoch_fname = 'epoch_d{0}_nc{1}_ne{2}.log'.format(model.depth,
                   model.num_class, num_epochs)

    short_fname = 'short_d{0}_nc{1}_ne{2}.log'.format(model.depth,
                  model.num_class, num_epochs)
    # Record loss every minute 
    prev_time = time.time()
    mins = 0
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):

        epoch_file = open(epoch_fname, 'a')
        epoch_file.write('Epoch {}/{}   ---   \n'.format(epoch+1, num_epochs))
        epoch_file.close()
 
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            epoch_start_time = time.time()
            running_loss = 0.0
            num_miss = 0
            cnt = 0
            samp = 0 
            # Iterate over data.
            (X, Y, batch_size) = dataloaders[phase]
            num_batches = int(len(X)/batch_size)
            over_flag = False # indicate if data are not evenly divided by batches            
            if num_batches*batch_size < len(X):
                over_flag = True
            for b in range(num_batches+1):
                if b == num_batches:
                    if over_flag:
                        inputs = torch.from_numpy(X[batch_size*b:]).to(device)
                        labels = torch.from_numpy(Y[batch_size*b:]).to(device)
                    else:
                        continue
                else:
                    inputs = torch.from_numpy(X[batch_size*b:batch_size*(b+1)]).to(device)
                    labels = torch.from_numpy(Y[batch_size*b:batch_size*(b+1)]).to(device)
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
                        num_miss += np.sum(yhat!=labels).item()
                        err_perc = num_miss/samp*100
                         
                if time.time()-prev_time >= 60:
                    mins += 1
                    short_file = open(short_fname, 'a')
                    if phase == 'train':
                        short_file.write('[{0}] {1} minute(s): loss = {2:.5f}\n'.format(phase,
                        mins, avg_loss))
                    if phase == 'test':
                        short_file.write('[{0}] {1} minute(s): error % = {2:.5f}\n'.format(
                        phase, mins, err_perc))
                    short_file.close()
                    prev_time = time.time()
                           
            if phase == 'train':
                epoch_loss = avg_loss
            if phase == 'test':
                epoch_loss = err_perc
            epoch_file = open(epoch_fname, 'a')
            epoch_file.write('[{0}] Loss: {1:.5f}\n'.format(phase, epoch_loss))
            epoch_file.close()
            print('*****[{0}] Time to complete epoch[{1}]: {2}'.format(phase, epoch, time.time()-epoch_start_time))

