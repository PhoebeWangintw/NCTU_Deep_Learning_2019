#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import dataloader

device = torch.device("cuda")

# ref: https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/pytorch/dataloader-and-datasets.html
# We have to override the methods __getitem__ and __len__ in torch.utils.data.Dataset parent class.
class LoadData(torch.utils.data.Dataset):
    def __init__(self, X_train, y_train):
        self.__xs = torch.from_numpy(X_train)
        self.__ys = torch.from_numpy(y_train)

    def __getitem__(self, index):
        return self.__xs[index], self.__ys[index]

    def __len__(self):
        return len(self.__xs)
    

def test_acc(model, loader):
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    return (correct / len(loader.dataset)) * 100 


class DeepConvNet(nn.Module):
    def __init__(self, activation_func='ELU', param=1.0):
        super(DeepConvNet, self).__init__()
        
        act_func = nn.ELU(alpha=param)
        if activation_func == 'ELU':
            act_func = nn.ELU(alpha=param)
        elif activation_func == 'ReLu':
            act_func = nn.ReLU(inplace=True)
        elif activation_func == 'LeakyReLu':
            act_func = nn.LeakyReLU(negative_slope=param, inplace=True)
        else:
            print("Cannot recognize activation function name.")
            act_func = nn.ELU(alpha=param)
        momentum=0.1
        dropout_rate=0.15

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=True),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            act_func,
            nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1, 1), padding=(1, 0), bias=True),
            nn.BatchNorm2d(25, eps=1e-05, momentum=momentum),
            act_func,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout_rate),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=True),
            nn.BatchNorm2d(50, eps=1e-05, momentum=momentum),
            act_func,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout_rate),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=True),
            nn.BatchNorm2d(100, eps=1e-05, momentum=momentum),
            act_func,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout_rate),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=True),
            nn.BatchNorm2d(200, eps=1e-05, momentum=momentum),
            act_func,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout_rate),
        )
        self.linear = nn.Sequential( 
            nn.Linear(200*3*46, 2, bias=True),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 200*3*46)
        x = self.linear(x)

        return F.log_softmax(x)


class EEGNet(nn.Module):
    def __init__(self, activation_func='ELU', param=1.0):
        super(EEGNet, self).__init__()
        act_func = nn.ELU(alpha=param)
        if activation_func == 'ELU':
            act_func = nn.ELU(alpha=param)
        elif activation_func == 'ReLu':
            act_func = nn.ReLU(inplace=True)
        elif activation_func == 'LeakyReLu':
            act_func = nn.LeakyReLU(negative_slope=param, inplace=True)
        else:
            print("Cannot recognize activation function name.")
            act_func = nn.ELU(alpha=param)
            
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 12), stride=(1, 1), padding=(0, 6), bias=True),
            nn.Conv2d(16, 16, kernel_size=(1, 25), stride=(1, 1), padding=(0, 12), bias=True),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            act_func,
        ) 
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=True),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            act_func,
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            act_func,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.20),
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=True),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            act_func,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.20),
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(-1, 32*23)
        x = self.classify(x)
        
        return x

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = dataloader.read_bci_data()
    print("Training data: ", X_train.shape)
    print("Training label: ", y_train.shape)
    print("Testing data: ", X_test.shape)
    print("Testing label: ", y_test.shape)
    print("-"*50)

    train = LoadData(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train, 
        batch_size=64,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
    test = LoadData(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    train = False
    
    if train:
        iteration = 800
        activation_func = ['ELU', 'ReLu', 'LeakyReLu']
#         params = [[1.0, 0.9, 0.8, 0.7], [0.0], [0.01, 0.02, 0.03, 0.04]]
#         params = [[0.9], [0.0], [0.03]]
        params = [[0.9], [0.0], [0.04]]

        acc_all = []
        train_acc_all = []
        loss_all = []
        max_acc = 0
        for idx, act_func in enumerate(activation_func):
            for param in params[idx]:
                print("current activation function: ", act_func)
                print("current param: ", param)
#                 model = EEGNet(act_func, param).to(device)
                model = DeepConvNet(act_func, param).to(device)
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                train_accs = []
                accs = []
                losses = []
                for epoch in range(iteration):
                    for data, target in train_loader:
                        data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)

                        optimizer.zero_grad()
                        output = model(data)
                        loss = F.cross_entropy(output, target)
                        loss.backward()
                        optimizer.step()
                    train_acc = test_acc(model, train_loader)
                    train_accs.append(train_acc)
                    acc = test_acc(model, test_loader)
                    if acc > max_acc:
                        print(acc)
#                         torch.save(model.state_dict(), "./EEGNet.pth")
                        torch.save(model.state_dict(), "./DeepConvNet.pth")
                        max_acc = acc
                    accs.append(acc)
                    if epoch % 100 == 0:
                        print(epoch)
                        print(acc)
                        print('-'*30)

                train_acc_all.append(train_accs)
                acc_all.append(accs)
                print(max(accs))
                print("-"*50)
        
        plt.figure(figsize=(15,10))
#         plt.title("EEGNet Result")
        plt.title("DeepConvNet")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        l1, = plt.plot(train_acc_all[0], linestyle='--', label='ELU: alpha=0.9(train)')
        print("ELU: alpha=0.9(train):", max(train_acc_all[0]))
        l2, = plt.plot(acc_all[0], label='ELU: alpha=0.9(test)')
        print("ELU: alpha=0.9(test):", max(acc_all[0]))
        l3, = plt.plot(train_acc_all[1], linestyle='--', label='ReLu(train)')
        print('ReLu(train):', max(train_acc_all[1]))
        l4, = plt.plot(acc_all[1], label='ReLu(test)')
        print('ReLu(test):', max(acc_all[1]))
#         l5, = plt.plot(train_acc_all[2], linestyle='--', label='LeakyReLu: neg_slope=0.03(train)')
#         print('LeakyReLu: neg_slope=0.03(train):', max(train_acc_all[2]))
#         l6, = plt.plot(acc_all[2], label='LeakyReLu: neg_slope=0.03(test)')
#         print('LeakyReLu: neg_slope=0.03(test):', max(acc_all[2]))
        l5, = plt.plot(train_acc_all[2], linestyle='--', label='LeakyReLu: neg_slope=0.04(train)')
        print('LeakyReLu: neg_slope=0.04(train):', max(train_acc_all[2]))
        l6, = plt.plot(acc_all[2], label='LeakyReLu: neg_slope=0.04(test)')
        print('LeakyReLu: neg_slope=0.04(test):', max(acc_all[2]))
        plt.legend(loc='lower right')
#         plt.savefig("EEGNet_result.png")
        plt.savefig("DeepConvNet_result.png")
        plt.clf()
        plt.cla()
        plt.close()
        
    else:
        activation_func = ['ELU', 'ReLu', 'LeakyReLu']
        params = [0.9, 0.0, 0.03]
        print("Accuracy during testing: ")
        print(EEGNet(activation_func[0], params[0]).to(device))
        for idx, act_func in enumerate(activation_func):
            model = EEGNet(act_func, params[idx]).to(device)
            model.load_state_dict(torch.load('./EEGNet.pth'))
            acc = test_acc(model, test_loader)
            print(act_func, "(", params[idx], "):", acc)
                
        activation_func = ['ELU', 'ReLu', 'LeakyReLu']
        params = [0.9, 0.0, 0.04]
        print("Accuracy during testing: ")
        print(DeepConvNet(activation_func[2], params[2]).to(device))
        for func in act_func:
            for param in params:        
                model = DeepConvNet(act_func, param).to(device)
                model.load_state_dict(torch.load('./DeepConvNet.pth'))
                acc = test_acc(model, test_loader)


    