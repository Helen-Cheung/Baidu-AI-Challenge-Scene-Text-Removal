import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from paddle.vision.transforms import Resize

from .networks import ConvWithActivation, DeConvWithActivation

def region_MS(img_org, img_out, mask):
    
    img_out = mask*img_out + (1-mask)*img_org

    return img_out

class Residual(nn.Layer):
    def __init__(self, in_channels, out_channels, same_shape=True, **kwargs):
        super(Residual, self).__init__()
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2D(in_channels, in_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2D(in_channels, out_channels, kernel_size=3, padding=1)
        if not same_shape:
            self.conv3 = nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=strides)
        self.batch_norm2d = nn.BatchNorm2D(out_channels)
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        out = self.batch_norm2d(out + x)
        return F.relu(out)

class PSPModule(nn.Layer):
    def __init__(self, features, out_features, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.LayerList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2D(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2D(output_size=(size, size))
        conv = nn.Conv2D(features, features, kernel_size=1, bias_attr=None)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = 16, 16
        priors = [F.upsample(stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(paddle.concat(priors, 1))
        return self.relu(bottle)

class PERT(nn.Layer):
    def __init__(self, n_inchannel=3):
        super(PERT, self).__init__()
        ###Backbone###
        self.conv1 = ConvWithActivation(n_inchannel*2, 32, kernel_size=4, stride=2, padding=1)        #256,256
        self.conv2 = ConvWithActivation(32, 64, kernel_size=4, stride=2, padding=1)       #128,128

        self.res1 = Residual(64, 64)
        self.res2 = Residual(64, 128, same_shape=False)                                   #64,64
        self.res3 = Residual(128, 128)
        self.res4 = Residual(128, 256, same_shape=False)                                  #32,32
        self.res5 = Residual(256, 256)
        self.res6 = Residual(256, 512, same_shape=False)                                  #16,16

        ###BRN_upsample###
        self.brn_deconv1 = DeConvWithActivation(512, 256, kernel_size=3, padding=1, stride=2, output_padding=1) #32,32
        self.brn_deconv2 = DeConvWithActivation(256*2, 128, kernel_size=3, padding=1, stride=2)                 #64,64
        self.brn_deconv3 = DeConvWithActivation(128*2, 64, kernel_size=3, padding=1, stride=2)                  #128,128
        self.brn_deconv4 = DeConvWithActivation(64*2,32,kernel_size=3,padding=1,stride=2)                       #256,256
        self.brn_deconv5 = DeConvWithActivation(64,3,kernel_size=3,padding=1,stride=2)                          #512,512

        ###TLN_upsample###
        self.psp = PSPModule(512,512)
        self.tln_deconv1 = DeConvWithActivation(512, 256, kernel_size=3, padding=1, stride=2)                   #32,32
        self.tln_deconv2 = DeConvWithActivation(256*2, 128, kernel_size=3, padding=1, stride=2)                 #64,64 
        self.tln_deconv3 = DeConvWithActivation(128*2, 64, kernel_size=3, padding=1, stride=2)                  #128,128
        self.tln_deconv4 = DeConvWithActivation(64, 64, kernel_size=3, padding=1, stride=2)                     #256,256
        self.tln_deconv5 = DeConvWithActivation(64, 64, kernel_size=3, padding=1, stride=2)                     #512,512
        self.mask_out = nn.Conv2D(64,3,kernel_size=1)                                                           #128,128
        self.sigmoid = nn.Sigmoid()

        ###MRM###
        self.conv3x3_p1 = ConvWithActivation(64, 3, kernel_size=3, stride=1, padding=1)                         #128,128
        self.conv3x3_p2 = ConvWithActivation(32, 3, kernel_size=3, stride=1, padding=1)                         #256,256
        

    def forward(self, img_org, img_out1):
        ###backbone###
        img = paddle.concat([img_org, img_out1], axis=1)
        x = self.conv1(img)          #256x256
        con_x1 = x
        x = self.conv2(x)               #128x128
        con_x2 = x
        x = self.res1(x)
        x = self.res2(x)                #64x64
        con_x3 = x
        x = self.res3(x)
        x = self.res4(x)                #32x32
        con_x4 = x
        x = self.res5(x)
        x = self.res6(x)                #16x16
        
        ###TLN###
        x_tln = self.psp(x)
        x_tln = self.tln_deconv1(x_tln)          #32x32
        x_tln = paddle.concat([x_tln, con_x4],axis=1) 
        x_tln = self.tln_deconv2(x_tln)          #64x64
        x_tln = paddle.concat([x_tln, con_x3],axis=1)
        x_tln = self.tln_deconv3(x_tln)          #128x128
        x_tln = self.tln_deconv4(x_tln)          #256x256
        x_tln = self.tln_deconv5(x_tln)          #512x512
        mask = self.mask_out(x_tln)
        mask_out = self.sigmoid(mask)

        ###BRN###
        x_brn = self.brn_deconv1(x)                  #32x32
        x_brn = paddle.concat([x_brn, con_x4],axis=1)
        x_brn = self.brn_deconv2(x_brn)              #64x64
        x_brn = paddle.concat([x_brn, con_x3],axis=1)
        x_brn = self.brn_deconv3(x_brn)              #128x128
        mrm_out1 = x_brn
        x_brn = paddle.concat([x_brn, con_x2],axis=1)    
        x_brn = self.brn_deconv4(x_brn)              #256x256
        mrm_out2 = x_brn
        x_brn = paddle.concat([x_brn, con_x1],axis=1)
        img_out = self.brn_deconv5(x_brn)              #512x512 
        

        ###MRM###
        p1_out = self.conv3x3_p1(mrm_out1)
        p2_out = self.conv3x3_p2(mrm_out2)

        img1_org = F.interpolate(img_org, scale_factor=0.25,mode='bilinear')
        img2_org = F.interpolate(img_org, scale_factor=0.5,mode='bilinear')

        mask1_out = F.interpolate(mask_out, scale_factor=0.25,mode='nearest')
        mask2_out = F.interpolate(mask_out, scale_factor=0.5,mode='nearest')


        img_out = region_MS(img_org, img_out, mask_out)
        p1_out = region_MS(img1_org, p1_out, mask1_out)
        p2_out = region_MS(img2_org, p2_out, mask2_out)
        

        return img_out, mask_out, p1_out, p2_out










