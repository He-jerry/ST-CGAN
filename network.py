from resnest.torch import resnest50
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from cbam import CBAM as CBAM

class fen(nn.Module):
  def __init__(self):
    super(fen,self).__init__()
    net=resnest50(pretrained=True)
    netlist=list(net.children())
    self.fe1=nn.Sequential(*netlist[0:4])#64
    self.fe2=nn.Sequential(*netlist[4])#256
    self.fe3=nn.Sequential(*netlist[5])#512
    self.fe4=nn.Sequential(*netlist[6])#1024
    self.fe5=nn.Sequential(*netlist[7])#2048
  def forward(self,x):
    fe1=self.fe1(x)
    fe2=self.fe2(fe1)
    fe3=self.fe3(fe2)
    fe4=self.fe4(fe3)
    fe5=self.fe5(fe4)
    return fe1,fe2,fe3,fe4,fe5
    
class residual(nn.Module):
	def __init__(self, channel_num):
		super(residual, self).__init__()
		self.conv_block1 = nn.Sequential(
			nn.Conv2d(channel_num, channel_num, 3, padding=1),
			nn.BatchNorm2d(channel_num),
			nn.ReLU(),
		) 
		self.conv_block2 = nn.Sequential(
			nn.Conv2d(channel_num, channel_num, 3, padding=1),
			nn.BatchNorm2d(channel_num),
		)
		self.relu = nn.ReLU()
	
	def forward(self, x):
		residual = x
		x = self.conv_block1(x)
		x = self.conv_block2(x)
		x = x + residual
		out = self.relu(x)
		return out
   
class residualcbam(nn.Module):
  def __init__(self,in_c):
    super(residualcbam,self).__init__()
    self.res=residual(in_c)
    self.cbam=CBAM(in_c)
  def forward(self,x):
    return self.cbam(self.res(x))
    
class residualse(nn.Module):
  def __init__(self,in_c):
    super(residualse,self).__init__()
    self.res=residual(in_c)
    self.se=seblock(in_c)
  def forward(self,x):
    return self.se(self.res(x))
    
class seblock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(seblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
        
class basicconv(nn.Module):
  def __init__(self,in_c,out_c,ks=4,st=2,pd=1):
    super(basicconv,self).__init__()
    self.conv=nn.Conv2d(in_c,out_c,ks,st,pd)
    self.bn=nn.BatchNorm2d(out_c)
    self.relu=nn.LeakyReLU(0.2)
    self.se=seblock(out_c)
  def forward(self,x):
    xt=self.conv(x)
    return self.relu(self.se(self.bn(xt)))
    
class basicdeconv(nn.Module):
  def __init__(self,in_c,out_c,ks=4,st=2,pd=1):
    super(basicdeconv,self).__init__()
    self.conv=nn.ConvTranspose2d(in_c,out_c,ks,st,pd)
    self.bn=nn.BatchNorm2d(out_c)
    self.relu=nn.LeakyReLU(0.2)
    #self.se=seblock(out_c)
  def forward(self,x):
    #return self.se(self.relu(self.bn(self.conv)))
    xt=self.conv(x)
    return self.relu(self.bn(xt))
    
class CPFE(nn.Module):
    def __init__(self, in_channels,feature_layer=None, out_channels=128):
        super(CPFE, self).__init__()

        self.dil_rates = [3, 5, 7]
        self.in_channels=in_channels

        # Define layers
        self.conv_1_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.conv_dil_3 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[0], padding=self.dil_rates[0], bias=False)
        self.conv_dil_5 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[1], padding=self.dil_rates[1], bias=False)
        self.conv_dil_7 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[2], padding=self.dil_rates[2], bias=False)

        self.bn = nn.BatchNorm2d(out_channels*4)

    def forward(self, input_):
        # Extract features
        conv_1_1_feats = self.conv_1_1(input_)
        conv_dil_3_feats = self.conv_dil_3(input_)
        conv_dil_5_feats = self.conv_dil_5(input_)
        conv_dil_7_feats = self.conv_dil_7(input_)

        # Aggregate features
        concat_feats = torch.cat((conv_1_1_feats, conv_dil_3_feats, conv_dil_5_feats, conv_dil_7_feats), dim=1)
        bn_feats = F.relu(self.bn(concat_feats))

        return bn_feats
        
class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=9):
        super(SpatialAttention, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        pad = (self.kernel_size-1)//2  # Padding on one side for stride 1

        self.grp1_conv1k = nn.Conv2d(self.in_channels, self.in_channels//2, (1, self.kernel_size), padding=(0, pad))
        self.grp1_bn1 = nn.BatchNorm2d(self.in_channels//2)
        self.grp1_convk1 = nn.Conv2d(self.in_channels//2, 1, (self.kernel_size, 1), padding=(pad, 0))
        self.grp1_bn2 = nn.BatchNorm2d(1)

        self.grp2_convk1 = nn.Conv2d(self.in_channels, self.in_channels//2, (self.kernel_size, 1), padding=(pad, 0))
        self.grp2_bn1 = nn.BatchNorm2d(self.in_channels//2)
        self.grp2_conv1k = nn.Conv2d(self.in_channels//2, 1, (1, self.kernel_size), padding=(0, pad))
        self.grp2_bn2 = nn.BatchNorm2d(1)

    def forward(self, input_):
        # Generate Group 1 Features
        grp1_feats = self.grp1_conv1k(input_)
        grp1_feats = F.relu(self.grp1_bn1(grp1_feats))
        grp1_feats = self.grp1_convk1(grp1_feats)
        grp1_feats = F.relu(self.grp1_bn2(grp1_feats))

        # Generate Group 2 features
        grp2_feats = self.grp2_convk1(input_)
        grp2_feats = F.relu(self.grp2_bn1(grp2_feats))
        grp2_feats = self.grp2_conv1k(grp2_feats)
        grp2_feats = F.relu(self.grp2_bn2(grp2_feats))

        added_feats = torch.sigmoid(torch.add(grp1_feats, grp2_feats))
        added_feats = added_feats.expand_as(input_).clone()

        return added_feats
        
class generator1(nn.Module):
  #inchannel=3,outchannel=1,sod branch
  def __init__(self,in_c=3,out_c=1):
    super(generator1,self).__init__()
    #encoder
    self.ochannel=out_c
    self.enb1=basicconv(in_c,64)
    self.enb2=basicconv(64,128)
    self.enb3=basicconv(128,256)
    self.enb4=basicconv(256,512)

    self.cpfe=CPFE(512)
    #self.sa=SpatialAttention(512)#

    #decoder
    self.dec1=basicdeconv(512,512,3,1,1)#sa
    self.dec2=basicdeconv(1024,512)#dec1+enb4
    self.dec3=basicdeconv(768,256)#dec2+enb3
    self.dec4=basicdeconv(256+128,128)#dec3+enb2
    self.dec5=basicdeconv(128+64,64)
    self.dec6=nn.ConvTranspose2d(64,out_c,3,1,1)
    self.sig=nn.Sigmoid()
    self.tanh=nn.Tanh()
  def forward(self,x):
    enb1=self.enb1(x)
    #enb1=F.interpolate(enb1,scale_factor=0.5)
    #print(enb1.shape)
    enb2=self.enb2(enb1)
    #enb2=F.interpolate(enb2,scale_factor=0.5)
    #print(enb2.shape)
    enb3=self.enb3(enb2)
    #enb3=F.interpolate(enb3,scale_factor=0.5)
    print(enb3.shape)
    enb4=self.enb4(enb3)
    #enb4=F.interpolate(enb4,scale_factor=0.5)
    #print(enb4.shape)

    cpfe=self.cpfe(enb4)
    #print(cpfe.shape)

    dec1=self.dec1(cpfe)

    #cat dec1+enb4
    #enb4=F.interpolate(enb4,scale_factor=1)
    dec1ct=torch.cat([dec1,enb4],1)
    #print(dec1ct.shape)
    dec2=self.dec2(dec1ct)
    #dec2=F.interpolate(dec2,scale_factor=2)
    print(dec2.shape)

    #cat dec2+enb3
    #enb3=F.interpolate(enb3,scale_factor=1)
    dec2ct=torch.cat([dec2,enb3],1)
    print(dec2ct.shape)
    dec3=self.dec3(dec2ct)
    #dec3=F.interpolate(dec3,scale_factor=4)
    #print(dec3.shape)

    #cat dec3+enb2
    #=F.interpolate(enb2,scale_factor=1)
    dec3ct=torch.cat([dec3,enb2],1)
    #print(dec3ct.shape)
    dec4=self.dec4(dec3ct)
    #dec4=F.interpolate(dec4,scale_factor=4)
    #print(dec4.shape)

    #cat dec4+enb1
    #enb1=F.interpolate(enb1,scale_factor=0.5)
    dec4ct=torch.cat([dec4,enb1],1)
    #print(dec4ct.shape)
    dec5=self.dec5(dec4ct)
    #dec5=F.interpolate(dec5,scale_factor=4)
    #print(dec5.shape)

    dec6=self.dec6(dec5)
    #dec6=F.interpolate(dec6,scale_factor=4)
    #print(dec6.shape)

    if self.ochannel==1:
      return dec6
    else:
      return self.tanh(dec6)
      
class inconv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(inconv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x

class lrl_conv_bn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(lrl_conv_bn, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x
class lrl_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(lrl_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
        
class discriminator(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(discriminator, self).__init__()
        self.inc = inconv(in_ch, 64)
        self.conv_1 = lrl_conv_bn(64, 128)
        self.conv_2 = lrl_conv_bn(128, 256)
        self.conv_3 = lrl_conv_bn(256, 512)
        self.conv_4 = lrl_conv(512, out_ch)

    def forward(self, input):
        cv0 = self.inc(input)
        cv1 = self.conv_1(cv0)
        cv2 = self.conv_2(cv1)
        cv3 = self.conv_3(cv2)
        cv4 = self.conv_4(cv3)
        out = torch.sigmoid(cv4)
        return out
        
class ASPP(nn.Module):
    def __init__(self):
        super(ASPP, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(1280, 512, kernel_size=1) # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(512)

    def forward(self, feature_map):
        # (feature_map has shape (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet instead is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8))

        feature_map_h = feature_map.size()[2] # (== h/16)
        feature_map_w = feature_map.size()[3] # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map))) # (shape: (batch_size, 256, h/16, w/16))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map))) # (shape: (batch_size, 256, h/16, w/16))

        out_img = self.avg_pool(feature_map) # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img))) # (shape: (batch_size, 256, 1, 1))
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear") # (shape: (batch_size, 256, h/16, w/16))

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1) # (shape: (batch_size, 1280, h/16, w/16))
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out))) # (shape: (batch_size, 256, h/16, w/16))

        return out
        
class generator2(nn.Module):
  def __init__(self):
    super(generator2,self).__init__()

    #encoder
    self.enc1=nn.Conv2d(3+1,64,3,1,1)#384
    self.enc2=lrl_conv(64,128)#192
    self.enc3=lrl_conv(128,256)#96
    self.enc4=lrl_conv(256,512)#48
    self.relu=nn.LeakyReLU(negative_slope=0.2, inplace=True)
    self.aspp=ASPP()

    #residual chain
    self.rs1=residualse(512)
    self.rc1=residualcbam(512)

    self.rs2=residualse(512)
    self.rc2=residualcbam(512)
    
    self.rs3=residualse(512)
    self.rc3=residualcbam(512)

    self.rs4=residualse(512)
    self.rc4=residualcbam(512)

    self.rs5=residualse(512)
    self.rc5=residualcbam(512)

    #decoder
    self.dec1=basicconv(512,256)
    self.dec2=basicconv(256,128)
    self.dec3=basicconv(128,64)
    self.dec4=nn.Conv2d(64,3,3,1,1)
    self.tanh=nn.Tanh()

  def forward(self,img,mask):
    inp=torch.cat([img,mask],1)
    enc1=self.enc1(inp)
    enc2=self.enc2(enc1)
    enc3=self.enc3(enc2)
    enc4=self.relu(self.enc4(enc3))
    aspp=self.aspp(enc4)

    rc=self.rc5(self.rs5(self.rc4(self.rs4(self.rc3(self.rs3(self.rc2(self.rs2(self.rc1(self.rs1(aspp))))))))))

    dec1=self.dec1(rc)
    dec2=self.dec2(dec1)
    dec3=self.dec3(dec2)
    dec4=self.dec4(dec3)
    dec4=F.interpolate(dec4,size=(img.shape[3],img.shape[2]))

    return self.tanh(dec4+img)