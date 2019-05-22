import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.utils.spectral_norm as spectral_norm
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
#         self.conv = nn.Sequential(
#                         nn.ConvTranspose2d(64, 512, kernel_size=(4, 4), stride=(1, 1), bias=False),
#                         nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
#                         nn.ReLU(inplace=True),
#                         nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
#                         nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
#                         nn.ReLU(inplace=True),
#                         nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
#                         nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
#                         nn.ReLU(inplace=True),
#                         nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
#                         nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
#                         nn.ReLU(inplace=True),
#                         nn.ConvTranspose2d(64, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
#                         nn.Tanh(),
#                     )
        self.fc = nn.Sequential(
            nn.Linear(64, 64*4*4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv = nn.Sequential( # 64x64x1x1
                    nn.Conv2d(64, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Upsample(scale_factor=2, mode='nearest'), # 8x8
                    nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Upsample(scale_factor=2, mode='nearest'), # 16x16
                    nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Upsample(scale_factor=2, mode='nearest'), # 32x32
                    nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Upsample(scale_factor=2, mode='nearest'), # 64x64
                    nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                    nn.Tanh(),
                )
        
    def forward(self, x):
        x = self.fc(x.view(-1, 64)).view(-1, 64, 4, 4)
        x = self.conv(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
#         self.conv = nn.Sequential(
#                             nn.Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
#                             nn.LeakyReLU(0.2, inplace=True),
#                             nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
#                             nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
#                             nn.LeakyReLU(0.2, inplace=True),
#                             nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
#                             nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
#                             nn.LeakyReLU(0.2, inplace=True),
#                             nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
#                             nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
#                             nn.LeakyReLU(0.2, inplace=True),
#                         )
        
#         self.d = nn.Sequential(
#                                 nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False),
#                                 nn.Sigmoid(),
#                             )
        
#         self.q = nn.Sequential(
#                             nn.Linear(8192, 100, bias=True),
#                             nn.ReLU(),
#                             nn.Linear(100, 10, bias=True),
#                         )
        
        self.conv = nn.Sequential(# (64, 64)
                            spectral_norm(nn.Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)),  # (32, 32)
                            nn.LeakyReLU(0.2, inplace=True),
                            spectral_norm(nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)), # (14, 14)
                            nn.LeakyReLU(0.2, inplace=True),
                            spectral_norm(nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)), # (7, 7)
                            nn.LeakyReLU(0.2, inplace=True),
                            spectral_norm(nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)), # (3, 3)
                            nn.LeakyReLU(0.2, inplace=True),
                        )
        self.d = nn.Sequential(
                                spectral_norm(nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), bias=False)),
                                nn.Sigmoid(),
                            )
        self.q = nn.Sequential(
                            nn.Linear(8192, 100, bias=True),
                            nn.ReLU(),
                            nn.Linear(100, 10, bias=True),
                            nn.Softmax(),
                        )

    def D(self, x):
        x = self.conv(x)
        x = self.d(x)
        
        return x.view(-1, 1)
        
    def Q(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.q(x)
        
        return x
        
    
    
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = dset.MNIST(root="./",
                                   download=True,
                                   transform=transforms.Compose([
                                       transforms.Resize(64, Image.BICUBIC),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,), (0.5,)),
                                   ]))
    
    lr_D = 2e-4 
    lr_G = 1e-3

#     lr_D = 4e-4 
#     lr_G = 1e-4

    iteration = 200
    dl_batch_size = 64
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dl_batch_size,
                                         shuffle=True, num_workers=4, pin_memory=True)
    
    fixed_noise = torch.randn(100, 54, 1, 1)
    row = torch.LongTensor(list(range(100)))
    col = torch.LongTensor(list(range(10))*10)
    one_hot_fix = torch.zeros(100, 10)
    one_hot_fix[row, col] = 1
    one_hot_fix = one_hot_fix.view(100, 10, 1, 1)
    fixed_noise = torch.cat((fixed_noise, one_hot_fix.float()), dim=1).to(device)
    
    bce_criterion = nn.BCELoss()   # categorical loss
    ce_criterion = nn.CrossEntropyLoss() 
   
    netG = Generator().to(device)
    netDQ = Discriminator().to(device)
    netD = netDQ.D
    netQ = netDQ.Q
            
    errGs = []
    errQs = []
    errDs = []
    D_judges = []
    
    optimizerD = optim.Adam([{'params': netDQ.conv.parameters()}, {'params': netDQ.d.parameters()}], lr=lr_D)
    optimizerG = optim.Adam([{'params': netG.parameters()}, {'params': netDQ.q.parameters()}], lr=lr_G)
    
    train = False

    if train:
        for epoch in tqdm(range(iteration)):
            errG_acc = 0
            errQ_acc = 0
            errD_acc = 0
            D_judge = []
            
            netG.train()
            netDQ.train()
            for data, _ in tqdm(dataloader, leave=False):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                
                optimizerD.zero_grad()

                batch_size = data.size(0)
                real_labels = torch.ones(batch_size, 1).to(device)
                fake_labels = torch.zeros(batch_size, 1).to(device)
                
                noise = torch.randn(batch_size, 54, 1, 1)  # 54d noise + 10d one_hot label
                rand_labels = torch.randint(low=0, high=10, size=(batch_size,))
                row = torch.LongTensor(list(range(batch_size)))
                one_hot = torch.zeros(batch_size, 10)
                one_hot[row, rand_labels] = 1
                noise = torch.cat((noise, one_hot.view(batch_size, 10, 1, 1)), dim=1).to(device)

                # train with real
                output_real = netD(data.to(device))
                errD_real = bce_criterion(output_real, real_labels)

                # train with fake
                fake_imgs = netG(noise)
                output_fake = netD(fake_imgs.detach())  # only update discriminator, don't backprop on netG
                errD_fake = bce_criterion(output_fake, fake_labels)

                errD = errD_real + errD_fake
                errD.backward()
                optimizerD.step()  # update discriminator
                
                errD_acc += errD


                ############################
                # (2) Update G network: maximize log(D(G(z))) + L(G, Q)
                ###########################
                optimizerG.zero_grad()
                
                output_fake = netD(fake_imgs)
                errG = bce_criterion(output_fake, real_labels)
                D_judge.extend(torch.round(output_fake))
                output_q = netQ(fake_imgs)
                errQ = ce_criterion(output_q, rand_labels.to(device))

                errGQ = errG + errQ
                errGQ.backward()
                optimizerG.step()

                errG_acc += errG
                errQ_acc += errQ

            errD_acc = errD_acc.detach().item()
            errQ_acc = errQ_acc.detach().item()
            errG_acc = errG_acc.detach().item()
            errDs.append(errD_acc)
            errQs.append(errQ_acc)
            errGs.append(errG_acc)
            mean = torch.mean(torch.FloatTensor(D_judge)).detach().item()
            D_judges.append(mean)

            if epoch % 1 == 0:
                print("epoch: %d: D loss: %.4f, G loss: %.4f, Q loss: %.4f, mean: %.4f" %(epoch, errD_acc, errG_acc, errQ_acc, mean))
                netG.eval()
                netDQ.eval()
                fake_img = netG(fixed_noise)
                torchvision.utils.save_image(torchvision.utils.make_grid(fake_img.detach(), nrow=10), 
                                                                './fake_img2/fake_samples_epoch_%03d.png' % (epoch),
                                                                normalize=True, range=(-1, 1))
                torch.save(errGs, "./loss2/" + str(epoch) + '_errG_losses.pt')
                torch.save(errQs, "./loss2/" + str(epoch) + '_errQ_losses.pt')
                torch.save(errDs, "./loss2/" + str(epoch) + '_errD_losses.pt')
                torch.save(D_judges, "./loss2/" + str(epoch) + '_D_judges.pt')

                torch.save(netG.state_dict(), './model_weight2/netG_epoch_%d.pth' % (epoch))
                torch.save(netDQ.state_dict(), './model_weight2/discriminator_epoch_%d.pth' % (epoch))
    else:
        epoch = 80
        
        netG.load_state_dict(torch.load('./model_weight2/netG_epoch_%d.pth' % (epoch)))
        netDQ.load_state_dict(torch.load('./model_weight2/discriminator_epoch_%d.pth' % (epoch)))
        
        netG.eval()
        netDQ.eval()
        fake_img = netG(fixed_noise)
        torchvision.utils.save_image(torchvision.utils.make_grid(fake_img.detach(), nrow=10), 
                                                                './fake_img2/test.png',
                                                                normalize=True, range=(-1, 1))
