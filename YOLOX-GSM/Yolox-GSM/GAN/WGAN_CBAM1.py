import torch
import torch.nn as nn

import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable

from GAN.CBAM import CBAM
# from thop import profile
# from thop import clever_format
class Generator(nn.Module):
    def __init__(self, channels=256):
        super(Generator, self).__init__()

        def downsample(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        def upsample(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers

        def Conv(in_feat,out_feat,normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 3, stride=1, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ReLU())
            return layers
            pass


        # self.model = nn.Sequential(
        #     *downsample(channels, 64, normalize=False),
        #     *downsample(64, 64),
        #     *downsample(64, 128),
        #     *downsample(128, 256),
        #     *downsample(256, 512),
        #     nn.Conv2d(512, 4000, 1),
        #     *upsample(4000, 512),
        #     *upsample(512, 256),
        #     *upsample(256, 128),
        #     *upsample(128, 64),
        #     *upsample(64, 32),
        #     *upsample(32, 32),
        #     nn.Conv2d(32, int(channels/2), 3, 1, 1),
        #
        #     nn.Tanh()
        # )
        # self.model = nn.Sequential(
        #     *downsample(channels, 128, normalize=False),
        #     *downsample(128, 64),
        #     *downsample(64, 32),
        #     # *downsample(128, 256),
        #     # *downsample(256, 512),
        #     nn.Conv2d(32, 32, 1),
        #     *upsample(32, 64),
        #     *upsample(64, 128),
        #     *upsample(128, 256),
        #     *upsample(256, 256),
        #     # *upsample(64, 32),
        #     # *upsample(32, 32),
        #     nn.Conv2d(256, int(channels / 2), 3, 1, 1),
        #
        #     nn.Tanh()
        # )
        self.cbam1 = CBAM(channels)
        self.downsample = nn.Sequential(
            *downsample(channels, 256, normalize=False),
            *Conv(256,128),
            *Conv(128,64),
            *Conv(64,32)
            # *downsample(128, 64),
            # *downsample(64, 32),
        )
        self.cbam2 = CBAM(32)
        self.upsample = nn.Sequential(
            # *upsample(32, 64),
            # *upsample(64, 64),
            *Conv(32,64),
            *upsample(64, 128),
            *upsample(128, 256),
            *upsample(256, 256)
        )
        self.cbam3 = CBAM(256)
        self.conv = nn.Sequential(
            *Conv(256,512),
            nn.Conv2d(512, int(channels / 2), 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.cbam1(x)
        x = self.downsample(x)
        x = self.cbam2(x)
        x = self.upsample(x)
        x = self.cbam3(x)
        x = self.conv(x)
        return x

class Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(

            #  256*128*128 ---> 256*64*64
            nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2,inplace=True),
            #  256*64*64 ---> 256*32*32
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),


            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # # State (256x16x16)
            # nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(512, affine=True),
            # nn.LeakyReLU(0.2, inplace=True),
            #
            # # State (512x8x8)
            # nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(1024, affine=True),
            # nn.LeakyReLU(0.2, inplace=True)
        )
            # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )


    def forward(self, x):
        # x = self.main_module(x)
        # x = self.output(x)
        # return x.view(-1)
        # print(x.shape)
        x = self.main_module(x)
        return self.output(x)


    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     inputs = torch.ones(1, 128, 160, 160)
#     inputs = inputs.to(device)
#     model = Discriminator(128)
#     output = model(inputs)
#     print(output.shape)
#     # model = DCGAN_D(128,256,64)
#
#     # model = model.to(device)
#     # output = model(inputs)
#     # print(output.shape)
#     # print(output.mean().shape)
#     pass

if __name__ == '__main__':
    # inputs = torch.randn(1, 128, 80, 80)
    # # inputs = inputs.cuda(0)
    # G = Generator(128)
    # output = G(inputs)
    # print(output.shape)
    #
    # flops, params = profile(G, inputs=(inputs,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print('flops:', flops)
    # print('params:', params)
    # # output = G(inputs)
    # # print(output.shape)

    real = torch.randn(1,64,40,40)
    D = Discriminator(64)
    output = D(real)
    print(output.shape)

    pass

