import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models
from torchvision import datasets, transforms
from torch.autograd.function import once_differentiable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import random
from tqdm import tqdm
from PIL import Image
import PIL

sns.set_style("whitegrid")
device = torch.device("cuda")

# dataloader
def getData(mode, root):
    if mode == 'train':
        img = pd.read_csv(root + 'train_img.csv')
        label = pd.read_csv(root + 'train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv(root + 'test_img.csv')
        label = pd.read_csv(root + 'test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)

class RetinopathyLoader(torch.utils.data.Dataset):
    def __init__(self, file_root, img_root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.img_root = img_root
        self.file_root = file_root
        self.img_name, self.label = getData(mode, file_root)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        path = self.img_root + self.img_name[index] + '.jpeg'
        label = self.label[index]
        
        img = Image.open(path).convert('RGB')
        if self.mode == 'train':
            transform_method = transforms.Compose([
                transforms.RandomCrop(480),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        elif self.mode == 'test':
            transform_method = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            print("invalid mode!")

        return transform_method(img), label

class ResNetPretrain(nn.Module):
    def __init__(self, model_name=50, pretrained=False):
        super(ResNetPretrain, self).__init__()
        if model_name == 50:
            self.classify = nn.Linear(2048, 5)
        elif model_name == 18:
            self.classify = nn.Linear(512, 5)
            
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(model_name)](pretrained=pretrained)
        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        del pretrained_model

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)

        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, s):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel * self.expansion),
            nn.ReLU(),
        )
        
        self.downsample = None
        
        if s != 1 or in_channel != out_channel * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * self.expansion, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm2d(out_channel * self.expansion),
            )
            
    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out, inplace=True)
        
        return out

class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Sequential(  # output = 256
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2_0 = nn.Sequential(  # output=128
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2_1 = self.make_layer(64, 64, 3, 1)  # output=128
        self.conv3 = self.make_layer(64*self.expansion, 128, 4, 2)  # output=64
        self.conv4 = self.make_layer(128*self.expansion, 256, 6, 2)  # output=32
        self.conv5 = self.make_layer(256*self.expansion, 512, 3, 2)  # output=16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(512 * self.expansion, 5)
        )
        
    def make_layer(self, in_channel, out_channel, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(Bottleneck(in_channel, out_channel, s))
            in_channel = out_channel * self.expansion
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_0(x)
        x = self.conv2_1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        
        return x
        
class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, s):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        
        self.downsample = None
        if s != 1 or in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=s, bias=False),
                nn.BatchNorm2d(out_channel),
            )
        
    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out, inplace=True)

        return out
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(  # output = 256
            nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )
        self.conv2_0 = nn.Sequential(  # output=128
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2_1 = self.make_layer(64, 2, 1)  # output=128
        self.conv3 = self.make_layer(128, 2, 2)  # output=64
        self.conv4 = self.make_layer(256, 2, 2)  # output=32
        self.conv5 = self.make_layer(512, 2, 2)  # output=16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Linear(512, 5)  # 5 classes to classify
        )
        
    def make_layer(self, out_channel, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channel, out_channel, s))
            self.in_channel = out_channel
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_0(x)
        x = self.conv2_1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # x.size(0) == batch_size
        x = self.linear(x)
        
        return x

def cal_acc(model, loader):
    correct = 0
    preds = []
    targets = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            preds.extend(pred)
            targets.extend(target.view_as(pred))
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    return (correct / len(loader.dataset)) * 100, targets, preds


def plot_confusion_matrix(y_true, y_pred, title):
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = [0, 1, 2, 3, 4]
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(title + "_cfmatrx.png")
    plt.clf()
    plt.cla()
    plt.close()
    
    return ax


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    batch_size = 10
    train = RetinopathyLoader('./data/', './imgs/', 'train')
    train_loader = torch.utils.data.DataLoader(
        train, 
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )
    
    
    test = RetinopathyLoader('./data/', './imgs/', 'test')
    test_loader = torch.utils.data.DataLoader(
        test, 
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )
    
    to_train = False

    if to_train:
        model_names = ["Resnet18", "Resnet50", "Resnet18_pretrain", "Resnet50_pretrain"]
        load_models = [False, False, False, False]
#         model_names = ["Resnet50_pretrain_2", "Resnet50_2"]
#         model_names = ["Resnet18_pretrain_2", "Resnet18_2"]
        model_names = ["Resnet18_2"]
        load_models = [False]
        
        for idx, model_name in enumerate(model_names):
            print(model_name)
            if model_name == "Resnet18_2":
                model = ResNet18().to(device)
                if load_models[idx]:
                    model.load_state_dict(torch.load("./" + model_name + ".pth"))
                iteration = 15
            elif model_name == "Resnet50_2":
                model = ResNet50().to(device)
                if load_models[idx]:
                    model.load_state_dict(torch.load("./" + model_name + ".pth"))
                iteration = 8
            elif model_name == "Resnet18_pretrain_2":
                if load_models[idx]:
                    model = ResNetPretrain(18, pretrained=False).to(device)
                    model.load_state_dict(torch.load("./" + model_name + ".pth"))
                else:
                    model = ResNetPretrain(18, pretrained=True).to(device)
                iteration = 15

            elif model_name == "Resnet50_pretrain_2":
                if load_models[idx]:
                    model = ResNetPretrain(50, pretrained=False).to(device)
                    model.load_state_dict(torch.load("./" + model_name + ".pth"))
                else:
                    model = ResNetPretrain(50, pretrained=True).to(device)
                iteration = 8
            else:
                print("Error! Cannot recognize model name.")
            
            train_accs = []
            test_accs = []
            max_acc = 0
            model.train(mode=True)
            optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
            for epoch in range(iteration):
                print("epoch:", epoch)
                correct = 0
                for (data, target) in tqdm(train_loader):
                    data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
                    optimizer.zero_grad()
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
                train_acc = (correct / len(train_loader.dataset)) * 100
                print('train_acc: ', train_acc)
                train_accs.append(train_acc)
                model.train(mode=False)
                test_acc, targets, preds = cal_acc(model, test_loader)
                model.train(mode=True)
                if test_acc > max_acc:
                    max_acc = test_acc
                    torch.save(model.state_dict(), "./" + model_name + ".pth")
                print("test_acc:", test_acc)
                test_accs.append(test_acc)

            print(train_accs)
            print(test_accs)
            plt.plot(train_accs, label="train")
            plt.plot(test_accs, label="test")
            plt.title(model_name)
            plt.legend(loc='lower right')
            plt.savefig(model_name + "_result.png")
            plt.clf()
            plt.cla()
            plt.close()
    
    else:
        model_names = ["./Resnet18_2.pth", "./Resnet50_2.pth", "./Resnet18_pretrain_2.pth", "./Resnet50_pretrain_2.pth"]
        models = [ResNet18().to(device), ResNet50().to(device), ResNetPretrain(18, pretrained=False).to(device), ResNetPretrain(50, pretrained=False).to(device)]
#         model_names = ["./Resnet18.pth", "./Resnet18_pretrain.pth"]
#         models = [ResNet18().to(device), ResNetPretrain(18, pretrained=False).to(device)]
        print("Testing")
        for idx, name in enumerate(model_names):
            print(name[2:-6])
            model = models[idx]
            model.load_state_dict(torch.load(name))
            model.eval()
            acc, targets, preds = cal_acc(model, test_loader)
            targets = torch.stack(targets)
            preds = torch.stack(preds)
            plot_confusion_matrix(targets.cpu().numpy(), preds.cpu().numpy(), name[2:-6])
            
            print("model:", name, ", acc:", acc)
