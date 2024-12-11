import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from DEPTHCONV import DEPTHWISECONV
from LA.DEPTHCONV import DEPTHWISECONV
# import DEPTHCONV
from thop import profile
from thop import clever_format


# 盗梦空间网络结构Inception
class InceptionA(torch.nn.Module):
    def __init__(self, in_channels,out_channels):
        super(InceptionA, self).__init__()

        # # 第一个分支 1*1  输出通道数16
        # self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        #
        # # 第二个分支 输出通道数24
        # self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        # self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)
        #
        # # 第三个分支 输出通道数24
        # self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        # self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        # self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)
        #
        # # 第四个分支 输出通道数 24
        # self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)


        self.branch1x1 = DEPTHWISECONV(in_ch=in_channels,out_ch=16,k=1,s=1,p=0)


        self.branch5x5_1 = DEPTHWISECONV(in_ch=in_channels,out_ch=16,k=1,s=1,p=0)
        self.branch5x5_2 = DEPTHWISECONV(in_ch=16,out_ch=24,k=5,s=1,p=2)



        self.branch3x3_1 = DEPTHWISECONV(in_ch=in_channels,out_ch=16,k=1,s=1,p=0)
        self.branch3x3_2 = DEPTHWISECONV(in_ch=16,out_ch=24,k=3,p=1)
        self.branch3x3_3 = DEPTHWISECONV(in_ch=24,out_ch=24,k=3,p=1)


        self.branch_pool = DEPTHWISECONV(in_ch=in_channels,out_ch=24,k=1,p=0)

        self.cat_conv = DEPTHWISECONV(in_ch=88,out_ch=out_channels,k=1,s=1,p=0,norm=True)


    def forward(self, x):
        brach1x1 = self.branch1x1(x)

        brach5x5 = self.branch5x5_1(x)
        brach5x5 = self.branch5x5_2(brach5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        print(brach1x1.shape,brach5x5.shape,branch3x3.shape,branch_pool.shape)

        output = [brach1x1, brach5x5, branch3x3, branch_pool]
        output = torch.cat(output, dim=1)
        result = self.cat_conv(output)

        # return torch.cat(output, dim=1)
        return result
    pass


class InceptionB(torch.nn.Module):
    def __init__(self, in_channels,out_channels):
        super(InceptionB, self).__init__()

        # # 第一个分支 1*1  输出通道数16
        # self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        #
        # # 第二个分支 输出通道数24
        # self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        # self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding=2)
        #
        # # 第三个分支 输出通道数24
        # self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        # self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1)
        # self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding=1)
        #
        # # 第四个分支 输出通道数 24
        # self.branch_pool = torch.nn.Conv2d(in_channels, 24, kernel_size=1)


        self.branch1x1 = DEPTHWISECONV(in_ch=in_channels,out_ch=16,k=1,s=1,p=0)
        self.branch7x7 = DEPTHWISECONV(in_ch=16,out_ch=16,k=7,s=2,p=3)


        self.branch5x5_1 = DEPTHWISECONV(in_ch=in_channels,out_ch=24,k=1,s=1,p=0)



        self.branch3x3_1 = DEPTHWISECONV(in_ch=in_channels,out_ch=16,k=1,s=1,p=0)
        self.branch3x3_2 = DEPTHWISECONV(in_ch=16,out_ch=24,k=3,p=1)
        self.branch3x3_3 = DEPTHWISECONV(in_ch=24,out_ch=24,k=3,s=2,p=1)


        self.branch_pool = DEPTHWISECONV(in_ch=in_channels,out_ch=24,k=1,p=0)

        self.cat_conv = DEPTHWISECONV(in_ch=88,out_ch=out_channels,k=1,s=1,p=0,norm=True)


    def forward(self, x):
        brach1x1 = self.branch1x1(x)
        brach1x1 = self.branch7x7(brach1x1)

        brach5x5 = F.max_pool2d(x,kernel_size=3, stride=2, padding=1)
        brach5x5 = self.branch5x5_1(brach5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        output = [brach1x1, brach5x5, branch3x3, branch_pool]
        output = torch.cat(output, dim=1)
        result = self.cat_conv(output)

        # return torch.cat(output, dim=1)
        return result
    pass



if __name__ == '__main__':
    inputs = torch.randn(1, 32, 100, 100)

    # print(type(inputs))
    # temp = inputs.unsqueeze(dim=1)
    # print(temp.shape)
    # print(inputs)
    # print(torch.max(inputs,dim=1))

    # self_attention = ResBlock_CBAM(32,64,stride=2,downsampling=True)
    self_attention = InceptionA(32,64)
    out = self_attention(inputs)

    flops, params = profile(self_attention, inputs=(inputs,))
    flops, params = clever_format([flops, params], "%.3f")
    print(out.shape)
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))

    pass