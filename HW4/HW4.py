#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import expit
from tqdm import tqdm_notebook
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import functools
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

device = torch.device("cuda")
sns.set_style('whitegrid')


# In[2]:


def data_generater(num_data):
    x = np.random.randint(0, 127, (2, num_data))
    out = x[0] + x[1]

    xs_0 = []
    xs_1 = []
    outs = []
    for i in range(num_data):
        x0_str = format(x[0][i], '08b')[::-1]
        x1_str = format(x[1][i], '08b')[::-1]
        out_str = format(out[i], '08b')[::-1]
        
        x0_char = []
        x1_char = []
        out_char = []
        for j in range(8):
            x0_char.append(int(x0_str[j]))
            x1_char.append(int(x1_str[j]))
            out_char.append(int(out_str[j]))
        xs_0.append(x0_char)
        xs_1.append(x1_char)
        outs.append(out_char)
        
    xs = [xs_0, xs_1]
    return np.array(xs), np.array(outs)


# In[3]:


class NaiveRNN():
    def __init__(self, in_dim, out_dim, hidden_dim, binary):
        np.random.seed(2)
        self.W = np.random.uniform(-1, 1, (hidden_dim, hidden_dim))
        self.b = np.zeros((hidden_dim, 1))
        
        self.U = np.random.uniform(-1, 1, (hidden_dim, in_dim))
        self.V = np.random.uniform(-1, 1, (out_dim, hidden_dim))
        self.c = np.zeros((out_dim, 1))
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.binary = binary
    
    def sigmoid(self, x):
        return expit(x)

    def H(self, h):
        return np.diagflat(np.mean(1 - (h * h), axis=1))
    
    def cross_entropy(self, y_pred, y_real):
        y_pred = np.clip(y_pred, 1e-15, 1.0-1e-15)
        return np.sum(-(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))) / (2 * y_pred.shape[0])
    
    def deriv_cross_entropy(self, y_pred, y_real):
        y_pred = np.clip(y_pred, 1e-15, 1.0-1e-15) 
        return ((-y_real / y_pred) + (1 - y_real) / (1 - y_pred)) / (y_pred.shape[0])
    
    def deriv_sigmoid(self, x):
        return np.multiply(x, 1.0 - x)
    
    def forward(self, x):
        h_t = np.zeros((self.hidden_dim, x.shape[1]))
        self.h_init = h_t
        self.h = []
        self.o = []
        
        for i in range(self.binary):
            a = self.b + self.W @ h_t + self.U @ x[:, :, i]
            h_t = np.tanh(a)
            self.h.append(h_t)
            self.o.append(self.sigmoid(self.c + (self.V @ h_t)))

        return np.concatenate(self.o).transpose(1, 0)
    
    def backpropagation(self, targets, x, lr=0.1):
        self.lr = lr
        grad_U = np.zeros((self.hidden_dim, self.in_dim))
        grad_V = np.zeros((self.out_dim, self.hidden_dim))
        grad_W = np.zeros((self.hidden_dim, self.hidden_dim))
        grad_c = np.zeros((self.out_dim, 1))
        grad_b = np.zeros((self.hidden_dim, 1))

        dLdo = 0
        for t in range(self.binary)[::-1]:
            dLdy = self.deriv_cross_entropy(self.o[t], targets[:, t]).reshape(1, -1)
            dLdo = dLdy * self.deriv_sigmoid(self.o[t])
            if t == self.binary - 1:
                dLdh = (self.V.T @ dLdo)
            else:
                dLdh = (self.V.T @ dLdo + self.W.T @ self.H(self.h[t+1]) @ dLdh)

            grad_V += (dLdo @ self.h[t].T)
            grad_c += np.sum(dLdo)
            grad_b += np.sum(self.H(self.h[t]) @ dLdh, axis=1).reshape(-1, 1)
            
            if t != 0:
                grad_W += (self.H(self.h[t]) @ dLdh @ self.h[t-1].T)
            grad_U += (self.H(self.h[t]) @ dLdh @ x[:, :, t].T)
            
        self.b = -grad_b * self.lr
        grad_c = -grad_c * self.lr
        grad_W = -grad_W * self.lr
        grad_V = -grad_V * self.lr
        grad_U = -grad_U * self.lr
        
        self.b += grad_b
        self.c += grad_c
        self.W += grad_W
        self.V += grad_V
        self.U += grad_U


# In[4]:


def prediction(outputs, targets):
    return (np.sum(np.abs(np.round(outputs) - targets)))

errs = []
losses = []
model = NaiveRNN(in_dim=2, out_dim=1, hidden_dim=16, binary=8)
lr = 0.008
for epoch in tqdm_notebook(range(20000)):
    xs, targets = data_generater(1)
    outputs = model.forward(xs)
    model.backpropagation(targets, xs, lr=lr)
    err = prediction(outputs, targets)
    errs.append(err)
    loss = model.cross_entropy(outputs, targets)
    losses.append(loss)
    
    if epoch % 100 == 0:
        print(epoch)
        print("loss: ", loss)
        print("err: ", err)

errs = np.array(errs)
losses = np.array(losses)
fig, ax = plt.subplots(nrows=2)
ax[0].plot(losses)
ax[1].plot(errs)
plt.show()


# In[5]:


print(np.sum(8 - errs[-1000:]) / 1000)


# In[6]:


plt.plot(8 - errs[-1000:])
plt.title("Accuracy of last 1000 iterations")
plt.savefig("last_1000.png")


# In[ ]:




