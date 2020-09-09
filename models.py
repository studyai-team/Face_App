import torch.nn as nn
import torch
from torchvision.models import resnet152, inception_v3
from torch.nn.functional import interpolate
import torch.nn.functional as F

from networks import IOLayer, DownsamplingLayer, UpsamplingLayer, ResBlock, SelfAttention, SPADE, NoiseLayer


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.inputlayer = IOLayer(3, 32)

        self.downsamplinglayer_1 = DownsamplingLayer(32, 64)
        self.downsamplinglayer_2 = DownsamplingLayer(64, 128)
        self.downsamplinglayer_3 = DownsamplingLayer(128, 256)
        self.downsamplinglayer_4 = DownsamplingLayer(256, 512)
        self.downsamplinglayer_5 = DownsamplingLayer(512, 512)

        self.upsamplinglayer_1 = UpsamplingLayer(512, 512)
        self.upsamplinglayer_2 = UpsamplingLayer(512, 256)
        self.self_attention_1 = SelfAttention(256)
        self.upsamplinglayer_3 = UpsamplingLayer(256, 128)
        self.self_attention_2 = SelfAttention(128)
        self.upsamplinglayer_4 = UpsamplingLayer(128, 64)
        self.upsamplinglayer_5 = UpsamplingLayer(64, 32)

        self.spade_1 = SPADE(512, 512)
        self.spade_2 = SPADE(512, 512)
        self.spade_3 = SPADE(256, 256)
        self.spade_4 = SPADE(128, 128)
        self.spade_5 = SPADE(64, 64)

        self.noise_1 = NoiseLayer(512, 8)
        self.noise_2 = NoiseLayer(512, 16)
        self.noise_3 = NoiseLayer(256, 32)
        self.noise_4 = NoiseLayer(128, 64)
        self.noise_5 = NoiseLayer(64, 128)

        self.outputlayer = IOLayer(32, 3)

    def forward(self, x, output_emb_1, output_emb_2, output_emb_3, output_emb_4, output_emb_5):
        x1 = self.inputlayer(x)

        x2 = self.downsamplinglayer_1(x1)
        x3 = self.downsamplinglayer_2(x2)
        x4 = self.downsamplinglayer_3(x3)
        x5 = self.downsamplinglayer_4(x4)
        x6 = self.downsamplinglayer_5(x5)

        x = self.spade_1(x6, output_emb_5)
        x = self.noise_1(x)
        x = torch.add(x, x6)
        x = self.upsamplinglayer_1(x)

        x = self.spade_2(x, output_emb_4)
        x = self.noise_2(x)
        x = torch.add(x, x5)
        x = self.upsamplinglayer_2(x)
        x = self.self_attention_1(x)

        x = self.spade_3(x, output_emb_3)
        x = self.noise_3(x)
        x = torch.add(x, x4)
        x = self.upsamplinglayer_3(x)
        x = self.self_attention_2(x)

        x = self.spade_4(x, output_emb_2)
        x = self.noise_4(x)
        x = torch.add(x, x3)
        x = self.upsamplinglayer_4(x)

        x = self.spade_5(x, output_emb_1)
        x = self.noise_5(x)
        x = torch.add(x, x2)
        x = self.upsamplinglayer_5(x)
        x = torch.add(x, x1)

        x = self.outputlayer(x)
        return x


class Embedder(nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()
        self.layer1 = ResBlock(3, 64)
        self.layer2 = ResBlock(64, 128)
        self.layer3 = ResBlock(128, 256)
        self.layer4 = ResBlock(256, 512)
        self.layer5 = ResBlock(512, 512)

    def forward(self, x):
        output_1 = self.layer1(x)
        output_2 = self.layer2(output_1)
        output_3 = self.layer3(output_2)
        output_4 = self.layer4(output_3)
        output_5 = self.layer5(output_4)
        return output_1, output_2, output_3, output_4, output_5


# pix2pix2-like discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
        )
        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
        )
        self.self_attention = SelfAttention(256)
        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
        )
        self.layer5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2)),
            nn.LeakyReLU(0.2),
        )
        self.layer6 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(512, 1, kernel_size=5, stride=1, padding=2)),
            nn.LeakyReLU(),
        )

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.self_attention(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


# Compressed network
class Generator64(nn.Module):
    def __init__(self):
        super(Generator64, self).__init__()
        self.inputlayer = IOLayer(3, 64)

        self.downsamplinglayer_1 = DownsamplingLayer(64, 128)
        self.downsamplinglayer_2 = DownsamplingLayer(128, 256)
        self.downsamplinglayer_3 = DownsamplingLayer(256, 512)

        self.upsamplinglayer_1 = UpsamplingLayer(512, 256)
        self.self_attention_1 = SelfAttention(256)
        self.upsamplinglayer_2 = UpsamplingLayer(256, 128)
        self.self_attention_2 = SelfAttention(128)
        self.upsamplinglayer_3 = UpsamplingLayer(128, 64)

        self.spade_1 = SPADE(512, 512)
        self.spade_2 = SPADE(256, 256)
        self.spade_3 = SPADE(128, 128)

        self.noise_1 = NoiseLayer(512, 8)
        self.noise_2 = NoiseLayer(256, 16)
        self.noise_3 = NoiseLayer(128, 32)

        self.outputlayer = IOLayer(64, 3)

    def forward(self, x, output_emb_1, output_emb_2, output_emb_3):
        x1 = self.inputlayer(x)

        x2 = self.downsamplinglayer_1(x1)
        x3 = self.downsamplinglayer_2(x2)
        x4 = self.downsamplinglayer_3(x3)

        x = self.spade_1(x4, output_emb_3)
        x = self.noise_1(x)
        x = torch.add(x, x4)
        x = self.upsamplinglayer_1(x)
        x = self.self_attention_1(x)

        x = self.spade_2(x, output_emb_2)
        x = self.noise_2(x)
        x = torch.add(x, x3)
        x = self.upsamplinglayer_2(x)
        x = self.self_attention_2(x)

        x = self.spade_3(x, output_emb_1)
        x = self.noise_3(x)
        x = torch.add(x, x2)
        x = self.upsamplinglayer_3(x)

        x = torch.add(x, x1)
        x = self.outputlayer(x)
        return x


class Embedder64(nn.Module):
    def __init__(self):
        super(Embedder64, self).__init__()
        self.layer1 = ResBlock(3, 128)
        self.layer2 = ResBlock(128, 256)
        self.layer3 = ResBlock(256, 512)

    def forward(self, x):
        output_1 = self.layer1(x)
        output_2 = self.layer2(output_1)
        output_3 = self.layer3(output_2)
        return output_1, output_2, output_3


# pix2pix2-like discriminator
class Discriminator64(nn.Module):
    def __init__(self):
        super(Discriminator64, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
        )
        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
        )
        self.self_attention = SelfAttention(256)
        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 1, kernel_size=5, stride=1, padding=2)),
            nn.LeakyReLU(),
        )

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.self_attention(out)
        out = self.layer4(out)
        return out


class GeneratorLight(nn.Module):
    def __init__(self):
        super(GeneratorLight, self).__init__()
        self.inputlayer = IOLayer(3, 32)

        self.downsamplinglayer_1 = DownsamplingLayer(32, 64)
        self.downsamplinglayer_2 = DownsamplingLayer(64, 128)
        self.downsamplinglayer_3 = DownsamplingLayer(128, 256)
        self.downsamplinglayer_4 = DownsamplingLayer(256, 512)
        self.downsamplinglayer_5 = DownsamplingLayer(512, 512)

        self.upsamplinglayer_1 = UpsamplingLayer(512, 512)
        self.upsamplinglayer_2 = UpsamplingLayer(512, 256)
        self.self_attention_1 = SelfAttention(256)
        self.upsamplinglayer_3 = UpsamplingLayer(256, 128)
        self.self_attention_2 = SelfAttention(128)
        self.upsamplinglayer_4 = UpsamplingLayer(128, 64)
        self.upsamplinglayer_5 = UpsamplingLayer(64, 32)

        self.spade_1 = SPADE(512, 512)
        self.spade_2 = SPADE(512, 512)
        self.spade_3 = SPADE(256, 256)
        self.spade_4 = SPADE(128, 128)
        self.spade_5 = SPADE(64, 64)

        self.outputlayer = IOLayer(32, 3)

    def forward(self, x, output_emb_1, output_emb_2, output_emb_3, output_emb_4, output_emb_5):
        x1 = self.inputlayer(x)

        x2 = self.downsamplinglayer_1(x1)
        x3 = self.downsamplinglayer_2(x2)
        x4 = self.downsamplinglayer_3(x3)
        x5 = self.downsamplinglayer_4(x4)
        x6 = self.downsamplinglayer_5(x5)

        x = self.spade_1(x6, output_emb_5)
        x = torch.add(x, x6)
        x = self.upsamplinglayer_1(x)

        x = self.spade_2(x, output_emb_4)
        x = torch.add(x, x5)
        x = self.upsamplinglayer_2(x)
        x = self.self_attention_1(x)

        x = self.spade_3(x, output_emb_3)
        x = torch.add(x, x4)
        x = self.upsamplinglayer_3(x)
        x = self.self_attention_2(x)

        x = self.spade_4(x, output_emb_2)
        x = torch.add(x, x3)
        x = self.upsamplinglayer_4(x)

        x = self.spade_5(x, output_emb_1)
        x = torch.add(x, x2)
        x = self.upsamplinglayer_5(x)
        x = torch.add(x, x1)

        x = self.outputlayer(x)
        return x
