#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 16:21:48 2021

@author: oyamatoshiki
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

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device:
    print('device is available!')

if not os.path.exists("save_image"):
    os.mkdir("save_image")
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
    
train_data = DownSizePairImageFolder("./woman-2", transform=transforms.ToTensor())
test_data = DownSizePairImageFolder("./woman-3", transform=transforms.ToTensor())
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
        
        self.pre_layer=nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3,64,kernel_size=9,stride=1,padding=4)),
            nn.PReLU()) #spectral_norm
        
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
if device:
    test_input=test_input.to(device)
    g=g.to(device)
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
if device:
    test_input=test_input.to(device)
    d=d.to(device)
out=d(test_input)
print(out.size())


def generator_loss(generated_image,hr_image,d_label,t_label):
    vgg=vgg16(pretrained=True) 
    content_layers=nn.Sequential(*list(vgg.features)[:31]).eval()
    if device:
        content_layers=content_layers.to(device)
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
D=Discriminator()

if device:
    G=G.to(device)
    D=D.to(device)
G_optimizer=optim.Adam(G.parameters(),lr=0.0001,betas=(0.9,0.999))
D_optimizer=optim.Adam(D.parameters(),lr=0.0001,betas=(0.9,0.999))

#学習率を変える。betaは原著論文の推奨通り変えない。

d_loss=nn.BCELoss()


def first_train(epoch):
    G.train()

    G_loss=0

    for batch_idx,(data_lr,data_hr)in enumerate(train_loader):
        if data_lr.size()[0]!=batch_size:
            break
        if device:
            data_lr=data_lr.to(device)
            data_hr=data_hr.to(device)
        fake_image=G(data_lr)
        G.zero_grad()

        G_loss=MSE_Loss(fake_image,data_hr)
        
        writer.add_scalar("first_Loss/train", G_loss, epoch) #lossの保存
        
        G_loss.backward()
        G_optimizer.step()
        G_loss+=G_loss.data.item()

        G_loss/=len(train_loader)

        g_image=fake_image.data.cpu()
        hr_image=data_hr.data.cpu()
        lr_image=data_lr.data.cpu()
        HR_image=torch.cat((hr_image, g_image),0)
        
        save_image(HR_image, "./save_image/epoch_{}.png".format(epoch))
        #save_image(hr_image, "./save_image/eopch_{}_hr.png".format(epoch))
        #save_image(lr_image, "./save_image/eopch_{}_lr.png".format(epoch))
        #save_image(g_image, "./save_image/eopch_{}_g.png".format(epoch))
        print("save_image_{}".format(epoch))

        return G_loss

num_epoch=10

for epoch in range(1,num_epoch+1):
    if epoch==1:
        print("trainning start!!")
    g_loss_=first_train(epoch)
    writer.flush()

    if epoch%1==0:
        torch.save(G.state_dict(),"./asset/G_first_epoch{}.pth".format(epoch))
    
writer.close()

'''

G=Generator(64)
D=Discriminator()

if device:
    G=G.to(device)
    D=D.to(device)

    g_param=torch.load("./asset/G_first_epoch{}.pth".format(epoch))
    G.load_state_dict(g_param)
else:
    g_param=torch.load("./asset/G_first_epoch{}.pth".format(epoch),map_location=lambda storage, loc:storage)
    G.load_state_dict(g_param)

    
G_optimizer=optim.Adam(G.parameters(),lr=0.0001,betas=(0.9,0.999))
D_optimizer=optim.Adam(D.parameters(),lr=0.0001,betas=(0.9,0.999))

d_loss=nn.BCELoss()




def train(epoch):
    D.train()
    G.train()

    y_real=torch.ones(batch_size,1)
    y_fake=torch.zeros(batch_size,1)

    if device:
        y_real=y_real.to(device)
        y_fake=y_fake.to(device)

        D_loss=0
        G_loss=0

    for batch_idx,(data_lr,data_hr)in enumerate(train_loader):
        if data_lr.size()[0]!=batch_size:
            break
        if device:
            data_lr=data_lr.to(device)
            data_hr=data_hr.to(device)
        print(batch_idx)
        D.zero_grad()
        G.zero_grad()

        D_real=D(data_hr)
        D_real_loss=d_loss(D_real,y_real)

        fake_image=G(data_lr)
        D_fake=D(fake_image)
        D_fake_loss=d_loss(D_fake,y_fake)

        D_loss=D_real_loss+D_fake_loss
        G_loss=generator_loss(fake_image,data_hr,D_fake,y_real)
        #G_loss=generator_loss(fake_image,data_hr,D_fake,y_fake)
        
        writer.add_scalar("G_Loss/train", G_loss, epoch) #G_lossの保存
        writer.add_scalar("D_Loss/train", D_loss, epoch) #D_lossの保存
        
        D_loss.backward(retain_graph=True)
        G_loss.backward()
        
        D_optimizer.step()
        G_optimizer.step()
        
        D_loss+=D_loss.data.item()
        G_loss+=G_loss.data.item()

        #G.zero_grad()

        #G_loss=generator_loss(fake_image,data_hr,D_fake,y_real)
        print(G_loss, D_loss)
        #G_loss.backward(requires_grad=True)
        #G_optimizer.step()
        #G_loss+=G_loss.data.item()

        D_loss/=len(train_loader)
        G_loss/=len(train_loader)

        if batch_idx%1==0:
            g_image=fake_image.data.cpu()
            hr_image=data_hr.data.cpu()
            HR_image=torch.cat((hr_image,g_image),0)
            save_image(HR_image,"./save_image/epoch_cont_{}.png".format(epoch))
            print("save_image")

        return D_loss,G_loss





num_epoch=10



for epoch in range(1,num_epoch+1):
    if epoch==1:
        print("trainning start!!")
    d_loss_,g_loss_=train(epoch)
    writer.flush()

    if epoch%40==0:
        #generate_image(epoch)
        torch.save(G.state_dict(),"./asset/G_2nd_epoch{}.pth".format(epoch))
        torch.save(D.state_dict(),"./asset/D_2nd_epoch{}.pth".format(epoch))


test_loader = DataLoader(test_data, batch_size, shuffle=True)
images_lr, images_hr = iter(test_loader).next()
generated_image=G(images_lr)
bl_recon=torch.nn.functional.upsample(images_lr,256,mode="bilinear")

save_image(torch.cat([images_hr,bl_recon.cpu(),generated_image.cpu()],0),"asset/result1.jpg")
      
writer.close()        
        
'''
test_loader = DataLoader(test_data, batch_size, shuffle=True)
images_lr, images_hr = iter(test_loader).next()
generated_image=G(images_lr)
bl_recon=torch.nn.functional.upsample(images_lr,256,mode="bilinear")  

save_image(torch.cat([images_hr,bl_recon.cpu(),generated_image.cpu()],0),"asset/result1.jpg")

#cd (ディレクトリ)
       
#tensorboard --logdir=runs
    
        
        
