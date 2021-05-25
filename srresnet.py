# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:15:21 2021

@author: Kumam
"""

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch
import os
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np
from torchvision.utils import save_image
#%matplotlib inline
from torchvision.models.vgg import vgg16

cuda = torch.cuda.is_available()
if cuda:
    print('cuda is available!')

if not os.path.exists("save_image_1"):
    os.mkdir("save_image_1")
if not os.path.exists("dataset"):
    os.mkdir("dataset")
if not os.path.exists("asset"):
    os.mkdir("asset")

class DownSizePairImageFolder(ImageFolder):
    def __init__(self, root, transform=None, large_size=256, large_size_2 = 256,
                 small_size=64, small_size_2 = 64, **kwds):
        super().__init__(root, transform=transform, **kwds)
        self.large_resizer = transforms.Resize((large_size, large_size_2))
        self.small_resizer = transforms.Resize((small_size, small_size_2))

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)
        large_img = self.large_resizer(img)
        small_img = self.small_resizer(img)
        if self.transform is not None:
            large_img = self.transform(large_img)
            small_img = self.transform(small_img)
        return small_img, large_img

train_data = DownSizePairImageFolder("./woman_train", transform=transforms.ToTensor())
test_data = DownSizePairImageFolder("./woman_test", transform=transforms.ToTensor())
batch_size = 8
train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=0)
print(train_data)
print(test_data)

images_lr, images_hr = iter(train_loader).next()
image=images_hr[0]
print(image.size())
image_np=image.numpy()
print(image_np.max(),image_np.min())
image_np=np.transpose(image_np,(1,2,0))
plt.imshow(image_np)

image=images_lr[0]
print(image.size())
image_np=image.numpy()
print(image_np.max(),image_np.min())
image_np=np.transpose(image_np,(1,2,0))
plt.imshow(image_np)

class Generator(nn.Module):
    def __init__(self,image_size):
        super(Generator,self).__init__()
        self.image_size=image_size

        self.pre_layer=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=9,stride=1,padding=4),
            nn.PReLU())

        self.residual_layer=nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64))

        self.middle_layer=nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64))

        self.pixcelshuffer_layer=nn.Sequential(
            Pixcelshuffer(64,2),
            Pixcelshuffer(64,2),
            nn.Conv2d(64,3,kernel_size=9,stride=1,padding=4))
    def forward(self,input_image):
        pre=self.pre_layer(input_image)
        res=self.residual_layer(pre)
        middle=self.middle_layer(res)
        middle=middle+pre
        output=self.pixcelshuffer_layer(middle)

        return output

class ResidualBlock(nn.Module):
    def __init__(self,input_channel):
        super(ResidualBlock,self).__init__()

        self.residualblock=nn.Sequential(
            nn.Conv2d(input_channel,input_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(input_channel),
            nn.PReLU(),
            nn.Conv2d(input_channel,input_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(input_channel))
    def forward(self,x):
        residual=self.residualblock(x)

        return x+residual

class Pixcelshuffer(nn.Module):
    def __init__(self,input_channel,r): #r=upscale_factor
        super(Pixcelshuffer,self).__init__()

        self.layer=nn.Sequential(
            nn.Conv2d(input_channel,256,kernel_size=3,stride=1,padding=1),
            nn.PixelShuffle(r),
            nn.PReLU())
    def forward(self,x):
        return self.layer(x)

test_input=torch.ones(1,3,64,64)
print(test_input.size())
g=Generator(64)
if cuda:
    test_input=test_input.cuda()
    g=g.cuda()
out=g(test_input)
print(out.size())

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.conv_layers=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))#(512,316,16)

        self.dense_layer=nn.Sequential(
            nn.Linear(16*16*512,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,1),
            nn.Sigmoid())

    def forward(self,input_image):
        batch_size=input_image.size()[0]

        conv=self.conv_layers(input_image)
        reshape=conv.view(batch_size,-1)
        output=self.dense_layer(reshape)

        return output


test_input=torch.ones(1,3,256,256)
d=Discriminator()
if cuda:
    test_input=test_input.cuda()
    d=d.cuda()
out=d(test_input)
print(out.size())

def generator_loss(generated_image,hr_image,d_label,t_label):
    vgg=vgg16(pretrained=True)
    content_layers=nn.Sequential(*list(vgg.features)[:31]).eval()
    if cuda:
        content_layers=content_layers.cuda()
    for param in content_layers.parameters():
        param.requires_grad=False
    mse_loss=nn.MSELoss()
    content_loss=mse_loss(content_layers(generated_image),content_layers(hr_image))

    BCE_loss=nn.BCELoss()
    adversarial_loss=BCE_loss(d_label,t_label)

    return content_loss+0.001*adversarial_loss

def MSE_Loss(generated_image,hr_image):
    mse_loss=nn.MSELoss()
    image_loss=mse_loss(generated_image,hr_image)

    return image_loss


G=Generator(64)
#D=Discriminator()

if cuda:
    G=G.cuda()
    #D=D.cuda()
G_optimizer=optim.Adam(G.parameters(),lr=0.0001,betas=(0.9,0.999))
#D_optimizer=optim.Adam(D.parameters(),lr=0.0001,betas=(0.9,0.999))

#d_loss=nn.BCELoss()

def first_train(epoch):
    G.train()

    G_loss=0

    for batch_idx,(data_lr,data_hr)in enumerate(train_loader):
        if data_lr.size()[0]!=batch_size:
            break
        if cuda:
            data_lr=data_lr.cuda()
            data_hr=data_hr.cuda()
        fake_image=G(data_lr)
        G.zero_grad()

        G_loss=MSE_Loss(fake_image,data_hr)
        G_loss.backward()
        G_optimizer.step()
        G_loss+=G_loss.data.item()

        G_loss/=len(train_loader)

        g_image=fake_image.data.cpu()
        hr_image=data_hr.data.cpu()
        lr_image=data_lr.data.cpu()
        HR_image=torch.cat((hr_image, g_image),0)
        save_image(HR_image,"./save_image_1/epoch_{}.png".format(epoch))
        print("save_image_{}".format(epoch))

        return G_loss

num_epoch=2000

'''
for epoch in range(1,num_epoch+1):
    if epoch==1:
        print("trainning start!!")
    g_loss_=first_train(epoch)

    if epoch%1==0:
        torch.save(G.state_dict(),"./asset/G_first_epoch{}.pth".format(epoch))
'''

def main():
    for epoch in range(1, num_epoch + 1):
        if epoch == 1:
            print("trainning start!!")
        g_loss_ = first_train(epoch)

        if epoch % 1 == 0:
            torch.save(G.state_dict(), "./asset/G_first_epoch{}.pth".format(epoch))


if __name__ == '__main__':
    main()
