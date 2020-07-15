import argparse
import os
import numpy
from skimage import io
import numpy as np
import math
import itertools
import sys
import torch
import torch.nn as nn
import torchvision
import os

import cv2
import torchvision.transforms as transforms
from torchvision.utils import save_image

#from network import totalnet
from dataset import ImageDataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform2 = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])
dtype = torch.FloatTensor
def tensor2im(input_image, imtype=np.uint8):
    mean = [0.5,0.5,0.5] 
    std = [0.5,0.5,0.5]  
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_img(im, path,h,w):
    im_grid = im
    im_numpy = tensor2im(im_grid) 
    im_array = Image.fromarray(im_numpy)
    im_array=im_array.resize((h,w))
    im_array.save(path)

with torch.no_grad():
  #net=generator()
  net1=torch.load("/public/zebanghe2/2020613/stgan2/generator1_ft_ 40.pth")
  net2=torch.load("/public/zebanghe2/2020613/stgan2/generator2_ft_ 40.pth")
net1.eval()
net1.cuda()
net2.eval()
net2.cuda()

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn
 
def save_output(image_name,pred,d_dir):

	predict = pred
	predict = predict.squeeze()
	predict_np = predict.cpu().data.numpy()

	im = Image.fromarray(predict_np*255).convert('RGB')
	img_name = image_name.split("/")[-1]
	image = io.imread(image_name)
	imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

	pb_np = np.array(imo)

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	imo.save(d_dir+imidx+'.png')


g = os.walk("/public/zebanghe2/joint/test/mix")  

for path,dir_list,file_list in g:  
    for file_name in file_list:
      tests=Image.open(path+'/'+file_name)
      #masks=cv2.imread("/public/zebanghe2/joint/test/sodmask"+'/'+file_name.split('.')[0]+'.png',cv2.IMREAD_GRAYSCALE)
      h,w=tests.size
      tests=transform(tests)
      #tests=tests.float()
      #masks=masks.astype(float)/255
      #masks=cv2.resize(masks,(384,384))
      #masks=transform2(masks)
      #masks=torch.from_numpy(masks)
      #masks=masks.float()
      #masks=masks.unsqueeze(0)
      #print(tests.shape)
      #print(masks.shape)
      #tests=torch.cat([tests,masks],0)
      tests=tests.unsqueeze(0)
      tests=tests.cuda()
      with torch.no_grad():
        fake_mask=net1(tests)
        #fake_in=torch.cat([tests,fake_mask],1)
        fake_B1=net2(tests,fake_mask)
        
      print(file_name)
      #mask1=faketrans.cpu().detach().numpy()[0,:,:,:]
      fake_B1=tensor2im(fake_B1.cpu()[0,:,:,:])
      save_img(fake_B1,"/public/zebanghe2/2020613/stgan2/ressult/"+file_name.split('.')[0]+'_bg.jpg',h,w)
      fake_B2=tensor2im(F.sigmoid(fake_mask).cpu()[0,:,:,:])
      #print(fake_B2)
      save_img(fake_B2,"/public/zebanghe2/2020613/stgan2/ressult/"+file_name.split('.')[0]+'_sod.png',h,w)
      #fake_B2=fake_B2.cpu().numpy()[0,0,:,:]
      #print(fake_B2)
      #fake_C=tensor2im(fake_C.cpu()[0,:,:,:])
      #save_img(fake_C,"/public/zebanghe2/derain/comparison/result/stcgan/"+file_name.split('.')[0]+'_trans.jpg',h,w)
      #mask1=denormalize(fake_B2.cpu().data[1])
      #mask1=denormalize(fake_B2.cpu().data[2])
      #sodmask=denormalize(fake_B2.cpu().data[0])
      #sodmask=fake_B2.cpu().numpy()[0,0,:,:]
      #smean=np.mean(sodmask)
      #sodmask[sodmask>smean]=255
      #sodmask[sodmask<smean]=0
      #print(sodmask)
      #sodmask=cv2.resize(sodmask,(h,w))
      #mask1=cv2.resize(mask1,(h,w))
      #cv2.imwrite("/public/zebanghe2/BASGan/output1/"+file_name.split('.')[0]+'_trans.jpg',mask1)
      #cv2.imwrite("/public/zebanghe2/526threestage/stcgan/output/"+file_name.split('.')[0]+'_ref2.png',sodmask)
