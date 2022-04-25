import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_


def downsample_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )


def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )


def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]


class SemDispNetS(nn.Module):

    def __init__(self, alpha=10, beta=0.01):
        super(SemDispNetS, self).__init__()

        self.alpha = alpha
        self.beta = beta

        convsize = [3, 32, 64, 128, 256, 512, 512, 512]
        
        # rgb encoding
        self.conv1 = downsample_conv(3,           convsize[0], kernel_size=7)
        self.conv2 = downsample_conv(convsize[0], convsize[1], kernel_size=5)
        self.conv3 = downsample_conv(convsize[1], convsize[2])
        self.conv4 = downsample_conv(convsize[2], convsize[3])
        self.conv5 = downsample_conv(convsize[3], convsize[4])
        self.conv6 = downsample_conv(convsize[4], convsize[5])
        self.conv7 = downsample_conv(convsize[5], convsize[6])
        
        upconvsize = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv(convsize[6],   upconvsize[0])
        self.upconv6 = upconv(upconvsize[0], upconvsize[1])
        self.upconv5 = upconv(upconvsize[1], upconvsize[2])
        self.upconv4 = upconv(upconvsize[2], upconvsize[3])
        self.upconv3 = upconv(upconvsize[3], upconvsize[4])
        self.upconv2 = upconv(upconvsize[4], upconvsize[5])
        self.upconv1 = upconv(upconvsize[5], upconvsize[6])
        
        # semantic encoding
        self.sconv1 = downsample_conv(3,           convsize[0], kernel_size=7)
        self.sconv2 = downsample_conv(convsize[0], convsize[1], kernel_size=5)
        self.sconv3 = downsample_conv(convsize[1], convsize[2])
        self.sconv4 = downsample_conv(convsize[2], convsize[3])
        self.sconv5 = downsample_conv(convsize[3], convsize[4])
        self.sconv6 = downsample_conv(convsize[4], convsize[5])
        self.sconv7 = downsample_conv(convsize[5], convsize[6])
        
        upconvsize = [512, 512, 256, 128, 64, 32, 16]
        self.upsconv7 = upconv(convsize[6],   upconvsize[0])
        # self.upsconv6 = upconv(upconvsize[0], upconvsize[1])
        # self.upsconv5 = upconv(upconvsize[1], upconvsize[2])
        # self.upsconv4 = upconv(upconvsize[2], upconvsize[3])
        # self.upsconv3 = upconv(upconvsize[3], upconvsize[4])
        # self.upsconv2 = upconv(upconvsize[4], upconvsize[5])
        # self.upsconv1 = upconv(upconvsize[5], upconvsize[6])


        self.iconv7 = conv(2 * (upconvsize[0] + convsize[5]), upconvsize[0])
        self.iconv6 = conv(upconvsize[1] + 2 * convsize[4], upconvsize[1])
        self.iconv5 = conv(upconvsize[2] + 2 * convsize[3], upconvsize[2])
        self.iconv4 = conv(upconvsize[3] + 2 * convsize[2], upconvsize[3])
        self.iconv3 = conv(1 + upconvsize[4] + 2 * convsize[1], upconvsize[4])
        self.iconv2 = conv(1 + upconvsize[5] + 2 * convsize[0], upconvsize[5])
        self.iconv1 = conv(1 + upconvsize[6], upconvsize[6])

        self.predict_disp4 = predict_disp(upconvsize[3])
        self.predict_disp3 = predict_disp(upconvsize[4])
        self.predict_disp2 = predict_disp(upconvsize[5])
        self.predict_disp1 = predict_disp(upconvsize[6])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x, sx = None):
        # import pdb; pdb.set_trace()
        
        # encode rgb 
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)
        
        # encode semantics
        out_sconv1 = self.conv1(sx)
        out_sconv2 = self.conv2(out_sconv1)
        out_sconv3 = self.conv3(out_sconv2)
        out_sconv4 = self.conv4(out_sconv3)
        out_sconv5 = self.conv5(out_sconv4)
        out_sconv6 = self.conv6(out_sconv5)
        out_sconv7 = self.conv7(out_sconv6)

        # up-convs
        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        out_upsconv7 = crop_like(self.upsconv7(out_sconv7), out_sconv6)
        concat7 = torch.cat((out_upconv7, out_upsconv7, out_conv6, out_sconv6), 1)
        out_iconv7 = self.iconv7(concat7)

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5, out_sconv5), 1)
        out_iconv6 = self.iconv6(concat6)

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4, out_sconv4), 1)
        out_iconv5 = self.iconv5(concat5)

        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3, out_sconv3), 1)
        out_iconv4 = self.iconv4(concat4)
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        disp4_up = crop_like(
            F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), 
            out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, out_sconv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(
            F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), 
            out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, out_sconv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(
            F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            return disp1
