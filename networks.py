import numpy as np
import torch
from torch import nn
from torch.nn.functional import elu, instance_norm
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import face_alignment
import random
import glob

import cv2
import os


class DownSamplingConvolutionLayer(nn.Module):
    def __init__(self, in_channel, out_channel, conv_size=3, padding_size=1):
        super(DownSamplingConvolutionLayer, self).__init__()

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.avg_pool2d = nn.AvgPool2d(2)

        # left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, ))

        # right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding=padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding=padding_size))

    def forward(self, x):
        res = x

        # left
        out_res = self.conv_l1(res)
        out_res = self.avg_pool2d(out_res)

        # right
        out = self.relu(x)
        out = self.conv_r1(out)
        out = self.relu_inplace(out)
        out = self.conv_r2(out)
        out = self.avg_pool2d(out)

        # merge
        out = out_res + out

        return out


class ResBlockDown(nn.Module):
    def __init__(self, in_channel, out_channel, conv_size=3, padding_size=1):
        super(ResBlockDown, self).__init__()

        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.avg_pool2d = nn.AvgPool2d(2)

        # left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, ))

        # right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding=padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding=padding_size))

    def forward(self, x):
        res = x

        # left
        out_res = self.conv_l1(res)
        out_res = self.avg_pool2d(out_res)

        # right
        out = self.relu(x)
        out = self.conv_r1(out)
        out = self.relu_inplace(out)
        out = self.conv_r2(out)
        out = self.avg_pool2d(out)

        # merge
        out = out_res + out

        return out


class IOLayer(nn.Module):
    # https://github.com/milesial/Pytorch-UNet/blob/67bf11b4db4c5f2891bd7e8e7f58bcde8ee2d2db/unet/unet_parts.py#L8
    def __init__(self, in_channels, out_channels):
        super(IOLayer, self).__init__()
        self.conv2d = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
            nn.ReLU())

    def forward(self, x):
        return self.conv2d(x)


class DownsamplingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingLayer, self).__init__()
        self.layer = nn.Sequential(nn.AvgPool2d(2),
                                   nn.utils.spectral_norm(
                                       nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
                                   nn.ReLU())

    def forward(self, x):
        return self.layer(x)


class UpsamplingLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)),
            nn.ReLU())

    def forward(self, x):
        return self.layer(x)


class ResBlock(nn.Module):
    # https://github.com/vincent-thevenin/Realistic-Neural-Talking-Head-Models/blob/master/network/blocks.py
    def __init__(self, in_channel, out_channel, conv_size=3, padding_size=1):
        super(ResBlock, self).__init__()

        self.relu = nn.ReLU()
        self.relu_inplace = nn.ReLU()
        self.avg_pool2d = nn.AvgPool2d(2)

        # left
        self.conv_l1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, 1, ))

        # right
        self.conv_r1 = nn.utils.spectral_norm(nn.Conv2d(in_channel, out_channel, conv_size, padding=padding_size))
        self.conv_r2 = nn.utils.spectral_norm(nn.Conv2d(out_channel, out_channel, conv_size, padding=padding_size))

    def forward(self, x):
        res = x

        # left
        out_res = self.conv_l1(res)
        out_res = self.avg_pool2d(out_res)

        # right
        out = self.relu(x)
        out = self.conv_r1(out)
        out = self.relu_inplace(out)
        out = self.conv_r2(out)
        out = self.avg_pool2d(out)

        # merge
        out = out_res + out

        return out


class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super(SelfAttention, self).__init__()

        # conv f
        self.conv_f = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 8, 1))
        # conv_g
        self.conv_g = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel // 8, 1))
        # conv_h
        self.conv_h = nn.utils.spectral_norm(nn.Conv2d(in_channel, in_channel, 1))

        self.softmax = nn.Softmax(-2)  # sum in column j = 1
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        f_projection = self.conv_f(x)  # BxC'xHxW, C'=C//8
        g_projection = self.conv_g(x)  # BxC'xHxW
        h_projection = self.conv_h(x)  # BxCxHxW

        f_projection = torch.transpose(f_projection.view(B, -1, H * W), 1, 2)  # BxNxC', N=H*W
        g_projection = g_projection.view(B, -1, H * W)  # BxC'xN
        h_projection = h_projection.view(B, -1, H * W)  # BxCxN

        attention_map = torch.bmm(f_projection, g_projection)  # BxNxN
        attention_map = self.softmax(attention_map)  # sum_i_N (A i,j) = 1

        # sum_i_N (A i,j) = 1 hence oj = (HxAj) is a weighted sum of input columns
        out = torch.bmm(h_projection, attention_map)  # BxCxN
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        ks = 3
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        if x.size()[2] != segmap.size()[2]:
            segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class NoiseLayer(nn.Module):
    def __init__(self, dim, size):
        super().__init__()

        self.fixed = False

        self.size = size
        self.register_buffer("fixed_noise", torch.randn([1, 1, size, size]))

        self.noise_scale = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        batch_size = x.size()[0]
        if self.fixed:
            noise = self.fixed_noise.expand(batch_size, -1, -1, -1)
        else:
            noise = torch.randn([batch_size, 1, self.size, self.size], dtype=x.dtype, device=x.device)
        return x + noise * self.noise_scale


# https://github.com/vincent-thevenin/Realistic-Neural-Talking-Head-Models
def generate_landmarks(frame):
    frame_landmark_list = []
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda:0')

    # for i in range(len(frames_list)):
    #     try:
    #         input = frames_list[i]
    preds = fa.get_landmarks(frame)[0]

    dpi = 100
    fig = plt.figure(figsize=(frame.shape[1] / dpi, frame.shape[0] / dpi), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(np.ones(frame.shape))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # chin
    ax.plot(preds[0:17, 0], preds[0:17, 1], marker='', markersize=5, linestyle='-', color='green', lw=2)
    # left and right eyebrow
    ax.plot(preds[17:22, 0], preds[17:22, 1], marker='', markersize=5, linestyle='-', color='orange', lw=2)
    ax.plot(preds[22:27, 0], preds[22:27, 1], marker='', markersize=5, linestyle='-', color='orange', lw=2)
    # nose
    ax.plot(preds[27:31, 0], preds[27:31, 1], marker='', markersize=5, linestyle='-', color='blue', lw=2)
    ax.plot(preds[31:36, 0], preds[31:36, 1], marker='', markersize=5, linestyle='-', color='blue', lw=2)
    # left and right eye
    ax.plot(preds[36:42, 0], preds[36:42, 1], marker='', markersize=5, linestyle='-', color='red', lw=2)
    ax.plot(preds[42:48, 0], preds[42:48, 1], marker='', markersize=5, linestyle='-', color='red', lw=2)
    # outer and inner lip
    ax.plot(preds[48:60, 0], preds[48:60, 1], marker='', markersize=5, linestyle='-', color='purple', lw=2)
    ax.plot(preds[60:68, 0], preds[60:68, 1], marker='', markersize=5, linestyle='-', color='pink', lw=2)
    ax.axis('off')

    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # frame_landmark_list.append((input, data))
    plt.close(fig)
    #     except:
    #         print('Error: Video corrupted or no landmarks visible')
    #
    # for i in range(len(frames_list) - len(frame_landmark_list)):
    #     # filling frame_landmark_list in case of error
    #     frame_landmark_list.append(frame_landmark_list[i])

    # return frame_landmark_list
    return data


# def select_frames(path, K):
#     file_list = random.shuffle(glob.glob(path))
#     if len(file_list) >= K:
#         file_list = file_list[:K]
#
#     frames_list = []
#     for file in file_list:
#         frame = cv2.imread(file)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames_list.append(frame)
#
#     return frames_list


def get_target_landmark_path(source_path):
    file_path = os.path.dirname(source_path)
    image_list = glob.glob(os.path.join(file_path, "*.jpg"))
    return random.choice(image_list)
