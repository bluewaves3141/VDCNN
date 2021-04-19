import torch
import torch.nn as nn
import sys
import numpy as np
import torch.optim as optim
import pandas as pd
import time

__all__ = ['Dataset', 'VDCNN', 'make_data',
           'make_dataloader', 'run_model']

KERNEL_SIZE = 3
EMBED_SIZE = 16
PADDING  = 1
MINUTE = 60

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
        # idxes = x.topk(self.k, dim=2)[1].sort(dim=0)[0]
        # return x.gather(2, idxes)
        return x.topk(self.k)[0]

class VDCNN(nn.Module):
    """
    Implmentation of VDCNN model by Conneau et al. 2017
    """
    def __init__(self, depth=9, num_class=10):
        super(VDCNN, self).__init__()
        self.depth = depth
        self.num_class = num_class
        
        self.embedding = nn.Embedding(73, EMBED_SIZE)
        
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
        self.conv64_block += [nn.Conv1d(EMBED_SIZE, 64, KERNEL_SIZE, padding=PADDING)]
        for i in range(num_blocks[0]):
            self.conv64_block += [nn.Conv1d(64, 64, KERNEL_SIZE, padding=PADDING), 
                                  nn.BatchNorm1d(64),
                                  nn.ReLU()]

        self.pool_half = nn.MaxPool1d(KERNEL_SIZE, stride=2, padding=PADDING) 
        
        self.conv128_block = nn.ModuleList()
        for i in range(num_blocks[1]):
            if i == 0:
                self.conv128_block += [nn.Conv1d(64, 128, KERNEL_SIZE, padding=PADDING)]
            else:
                self.conv128_block += [nn.Conv1d(128, 128, KERNEL_SIZE, padding=PADDING)]
            self.conv128_block += [nn.BatchNorm1d(128),
                                   nn.ReLU()]   
        
        self.conv256_block = nn.ModuleList()
        for i in range(num_blocks[2]):
            if i == 0:
                self.conv256_block += [nn.Conv1d(128, 256, KERNEL_SIZE, padding=PADDING)]
            else:
                self.conv256_block += [nn.Conv1d(256, 256, KERNEL_SIZE, padding=PADDING)]
            self.conv256_block += [nn.BatchNorm1d(256),
                                   nn.ReLU()]
        
        self.conv512_block = nn.ModuleList()
        for i in range(num_blocks[3]):
            if i == 0:
                self.conv512_block += [nn.Conv1d(256, 512, KERNEL_SIZE, padding=PADDING)]
            else:
                self.conv512_block += [nn.Conv1d(512, 512, KERNEL_SIZE, padding=PADDING)]
            self.conv512_block += [nn.BatchNorm1d(512),
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
    chrs = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"\\/|_@#$%^&*~`+-=<>()[]{}\n "
    for i, c in enumerate(chrs):
        lookup_table[c] = i+1 # reserve 0 as padding
    train_X = np.zeros([len(train_df), 1024]).astype(int)
    test_X = np.zeros([len(test_df), 1024]).astype(int)

    label_col = 0
    if dataset_name == 'yelp_review_polarity_csv':
        data_col = 1
    elif dataset_name == 'yahoo_answers_csv':
        data_col = 3 

    X = [train_X, test_X]
    dfs = [train_df, test_df]
    
    # phase 0 = train / phase 1 = test
    for phase in range(2):
        for i, text in enumerate(dfs[phase].loc[:, data_col]):
            for j, cc in enumerate(str(text)):           
                if cc not in lookup_table.keys():
                    X[phase][i, j] = 0
                else:
                    X[phase][i, j] = lookup_table[cc]
                # text longer than 1024 length
                if j == 1023:
                    break

    # account for target label starting at 1
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
    2 log files.
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
    criterion = nn.CrossEntropyLoss(reduction='sum')

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
            samp = 0 
            # Iterate over data.
            (X, Y, batch_size) = dataloaders[phase]
            num_batches = len(X) // batch_size
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

                        # added gradient clip to stabilize training
                        clip_norm = 3.0 # from cjiang2 github
                        nn.utils.clip_grad_norm(model.parameters(), clip_norm)

                        optimizer.step()
                        # statistics
                        running_loss += loss.item()
                        avg_loss = running_loss/samp
                    if phase == 'test':
                        yhat = torch.argmax(outputs, dim=1)
                        num_miss += (~yhat.eq(labels)).sum().item()
                        err_perc = num_miss/samp*100
                         
                if time.time()-prev_time >= MINUTE:
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