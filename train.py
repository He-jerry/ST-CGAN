#! /usr/bin/env python
import argparse
import os
import numpy as np
import math
import itertools
import sys
import torch
import torch.nn as nn
import torchvision
from loss import VGGLoss,localmse2
torch.cuda.empty_cache()
from dataset import ImageDataset
from network import generator1,generator2,discriminator
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
# Loss functions
#new code
criterion_GAN= torch.nn.MSELoss()
criterion_GAN2= torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()#L2 loss
criterion_bce=torch.nn.BCEWithLogitsLoss()
criterion_vgg=localmse2()

gen1 =generator1(3,1)
gen2=generator2()
discriminator1 = discriminator(4,1)
discriminator2 = discriminator(7,1)


gen1 = gen1.cuda()
gen2 = gen2.cuda()
discriminator1 = discriminator1.cuda()
discriminator2 = discriminator2.cuda()
criterion_GAN.cuda()
criterion_GAN2.cuda()
criterion_pixelwise.cuda()
criterion_bce.cuda()
criterion_vgg.cuda()

gen1.apply(weights_init_normal)
#gen2.apply(weights_init_normal)
discriminator1.apply(weights_init_normal)
discriminator2.apply(weights_init_normal)
# Optimizers
optimizer_G1 = torch.optim.Adam(gen1.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_G2 = torch.optim.Adam(gen2.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D1 = torch.optim.Adam(discriminator1.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D2 = torch.optim.Adam(discriminator2.parameters(), lr=0.0001, betas=(0.5, 0.999))
#optimizer_T = torch.optim.Adam(netparam, lr=0.00005, betas=(0.5, 0.999))
# Configure dataloaders


trainloader = DataLoader(
    ImageDataset(transforms_=None),
    batch_size=2,
    shuffle=False,drop_last=True
)
print("data length:",len(trainloader))
Tensor = torch.cuda.FloatTensor
eopchnum=70#10+40+10+40+60
print("start training")
for epoch in range(0, eopchnum):
  print("epoch:",epoch)
  iteration=0
  for i, total in enumerate(trainloader):
    
    iteration=iteration+1
    # Model inputs
    real_img = total["mix"]
    real_trans=total["trans"]
    real_mask = total["mask"]
    real_img=real_img.cuda()
    real_trans=real_trans.cuda()
    real_mask=real_mask.cuda()
    
    # Model inputs
    real_A = Variable(total["mix"].type(Tensor))
    real_B = Variable(total["trans"].type(Tensor))
    real_C = Variable(total["mask"].type(Tensor))

    # Adversarial ground truths
    valid = Variable(Tensor(np.ones((real_A.size(0), 1,12,12))), requires_grad=False)
    fake = Variable(Tensor(np.zeros((real_A.size(0), 1,12,12))), requires_grad=False)
    print("batch:%3d,iteration:%3d"%(epoch+1,iteration))
    # ------------------
    #  Train Generators1,2
    # ------------------

    optimizer_G1.zero_grad()
    # GAN loss
    outmap = gen1(real_A)
    gp1=torch.cat([real_A,outmap],1)
    pred_fake = discriminator1(gp1)
    loss_GAN = criterion_GAN(pred_fake, valid)
    #loss_GAN=Variable(loss_GAN,requires_grad=True)
    # Pixel-wise loss
    loss_pixel = criterion_bce(outmap, real_C)
    #loss_pixel=Variable(loss_pixel,requires_grad=True)
    loss_G1=loss_GAN+loss_pixel
    print("loss GAN:%3f,loss_pixel:%3f"%(loss_GAN,loss_pixel))
      
    loss_G1.backward(retain_graph=True)
    #for name, weight in gen1.named_parameters():
           #if weight.requires_grad:	
		          #train_iterator.set_description(name,str(parms.requires_grad),str(parms.grad))
             #print("name",name)
             #print(name,weight.grad.mean())

    optimizer_G1.step()
      
    optimizer_G2.zero_grad()
    # GAN loss
    outtrans = gen2(real_A,outmap)
    gp3=torch.cat([real_A,outmap,outtrans],1)
    pred_fake2 = discriminator2(gp3)
    loss_GAN2 = criterion_GAN(pred_fake2, valid)
    #loss_GAN2=Variable(loss_GAN2,requires_grad=True)
    # Pixel-wise loss
    loss_pixel2 = criterion_vgg(outtrans, real_B,real_C)
    #loss_pixel2=Variable(loss_pixel2,requires_grad=True)
    loss_G2=loss_GAN2+(loss_pixel2)
    print("loss GAN2:%3f,loss_pixel2:%3f"%(loss_GAN2,loss_pixel2))
      
    loss_G2.backward()

    optimizer_G2.step()
    #for name, weight in gen2.named_parameters():
           #if weight.requires_grad:	
		          #train_iterator.set_description(name,str(parms.requires_grad),str(parms.grad))
             #print("name",name)
             #print(name,weight.grad.mean())
    print("gen1:%3f,gen2:%3f"%(loss_G1.item(),loss_G2.item()))


    outmap=gen1(real_A)
    outtrans=gen2(real_A,outmap)
    
    # ---------------------
    #  Train Discriminator1,2
    # --------------------- 

    optimizer_D1.zero_grad()

    # Real loss
    tp1=torch.cat([real_A,real_C],1)
    pred_real = discriminator1(tp1)
    loss_real = criterion_GAN(pred_real, valid)
    #loss_real=Variable(loss_real,requires_grad=True)
      
    tk1=torch.cat([real_A,outmap],1)
    pred_fake = discriminator1(tk1)
    loss_fake = criterion_GAN(pred_fake.detach(), fake)
    #loss_fake=Variable(loss_fake,requires_grad=True)

    loss_D1 = loss_real+loss_fake

    loss_D1.backward(retain_graph=True)
    optimizer_D1.step()
    optimizer_D2.zero_grad()
    tp2=torch.cat([real_A,real_C,real_B],1)
    pred_real2 = discriminator2(tp2)
    loss_real2 = criterion_GAN2(pred_real2, valid)
    #loss_real2=Variable(loss_real2,requires_grad=True)
    tk2=torch.cat([real_A,outmap,outtrans],1)
    pred_fake2 = discriminator2(tk2)
    loss_fake2 = criterion_GAN2(pred_fake2.detach(), fake)
    #loss_fake2=Variable(loss_fake2,requires_grad=True)

    loss_D2 = loss_real2+loss_fake2

    loss_D2.backward()
    optimizer_D2.step()
    print("dis1:%3f,dis2:%3f"%(loss_D1.item(),loss_D2.item()))

    
    
    
    
  if(epoch%10==0):
    torch.save(gen1,"generator1_ft_%3d.pth"%epoch)
    torch.save(discriminator1,'discriminator1_ft_%3d.pth'%epoch)
    torch.save(gen2,"generator2_ft_%3d.pth"%epoch)
    torch.save(discriminator2,'discriminator2_ft_%3d.pth'%epoch)



    