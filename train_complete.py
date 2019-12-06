#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Luca Guastoni
"""

from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Network Parameters
HIDDEN_SIZE = 30
INPUT_LENGTH = 10

LR = 0.005
STEP_EPOCHS = 25
N_EPOCHS = 100
BATCH_SIZE = 32

# Training parameters
I_SERIES = 0		# 0 - 5,000 training samples
			# 1 - 50,000 training samples
TEST = True

#%% Network definition

class RNN_Net(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                 num_layers=1):
        super(RNN_Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(
                input_size = self.input_dim,
                hidden_size = self.hidden_dim,
                num_layers = self.num_layers,
                batch_first = True,
                nonlinearity='relu'
                )
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        
    def init_hidden(self):
        return torch.zeros(self.batch_size, self.num_layers, self.hidden_dim)
        
    def forward(self, x):
        out_rnn, self.hidden = self.rnn(x)
        output = self.linear(out_rnn[:,-1])
        return output
         
#%% Data pre-processing
def preprocess_data_recurrent(i_ser,p):
    
    """
    Args:
      i_ser:  Numerical value that indicate which timeseries has to be loaded
      p:      Number of elements in the initial sequence
    Returns:
      A numpy array with sequences of p elements, to be used as features 
      for the model
    """
    t_series = np.load(f'./.datasets/series_{i_ser:04d}.npz')['tseries']
    time_series = t_series
    T = time_series.shape[1]
    print(T)
    processed_features = np.ndarray((T-p,p,3),float)
    output_targets = np.ndarray((T-p,3),float)
     
    for i_p in range(0,T-p):    
        for i_coeff in range(3):
            
            processed_features[i_p,:,i_coeff] = time_series[i_coeff,i_p:i_p+p]
            output_targets[i_p,i_coeff] = time_series[i_coeff,i_p+p]
     
    return processed_features, output_targets     
            
if __name__ == '__main__':
    #load data
    p = INPUT_LENGTH
    
    X, Y =  preprocess_data_recurrent(I_SERIES,p)
    n_train = int(10**np.floor(np.log10(X.shape[0]))/2)
    n_val = n_train
    n_test = X.shape[0] - n_train*2

    X_train = torch.tensor(X[:n_train])
    Y_train = torch.tensor(Y[:n_train])
    
    X_val = torch.tensor(X[n_train:n_train+n_val])
    Y_val = torch.tensor(Y[n_train:n_train+n_val])
    
    X_test = torch.tensor(X[-n_test:])
    Y_test = torch.tensor(Y[-n_test:])

    #build the model
    net = RNN_Net(3, HIDDEN_SIZE, batch_size = BATCH_SIZE, 
                  output_dim = 3, num_layers = 1)
    net.double()
    print(net)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=STEP_EPOCHS, 
                                                gamma=0.5)
    

#%% Model training
    loss_vec = np.ndarray((2,N_EPOCHS), dtype='float')
    for i_t in range(N_EPOCHS):
        
        print(f'Epoch: {i_t+1}/{N_EPOCHS}')
        permutation = torch.randperm(n_train)        
        
        loop = tqdm(X_train, unit='samples')
        total_loss = 0
        for i in range(0,n_train, BATCH_SIZE):
            # Clear stored gradient
            optimizer.zero_grad()
            net.hidden = net.init_hidden()
            
            i_batch = min(i+BATCH_SIZE,n_train)
            indices = permutation[i:i_batch]
            batch_x, batch_y = X_train[indices], Y_train[indices]
            
            outputs = net.forward(batch_x)
            loss = criterion(outputs,batch_y)
            with torch.no_grad():
                total_loss = (total_loss*i + loss.detach().numpy()*(i_batch-i))/i_batch
                loop.set_postfix({'Training loss':f'{total_loss:.4e}'})
                loop.update(i_batch-i)
    
            loss.backward()
            optimizer.step()
        scheduler.step()
        loss_vec[0,i_t] = total_loss
        # Compute the validation loss
        with torch.no_grad():
            outputs = net.forward(X_val)
            val_loss = criterion(outputs,Y_val) 
        loss_vec[1,i_t] = val_loss.detach().numpy()
        
        loop.set_postfix({'Training loss':f'{total_loss:.4e}', 'Validation loss':f'{val_loss.detach().numpy():.4e}'})   
        loop.close()  

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(loss_vec[0,1:], label='Training')
ax.plot(loss_vec[1,1:], label='Validation')
ax.legend()
ax.set(xlabel='$Epoch$', ylabel='$Loss$')

if TEST == False:    
    sys.exit(0)

#%% Prediction
    
with torch.no_grad():
    total_test_loss = np.ndarray((n_test,), dtype='float')
    
    x_pr = X_test[0]
    for i_pred in range(0,n_test):
        x_p = x_pr[i_pred:p+i_pred].view((1,p,3))
        new_pred = net.forward(x_p)
        test_loss = criterion(new_pred,Y_test[i_pred].view((-1,3)))
        total_test_loss[i_pred] = test_loss.numpy()
        x_pr = torch.cat([x_pr,new_pred], dim=0)
    
    x_pred = x_pr.transpose(0,1)
    
x_pred_np = x_pred.detach().numpy()
X_test_np = X_test.detach().numpy()
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X_test_np[:100,0,0], X_test_np[:100,0,1], X_test_np[:100,0,2])
ax.plot(x_pred_np[0,:100], x_pred_np[1,:100], x_pred_np[2,:100])

plt.show()
