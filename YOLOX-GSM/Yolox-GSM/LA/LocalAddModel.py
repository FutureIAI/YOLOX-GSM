from LA.MHSA import MulHeadSelfAttention
from LA.MSA import MulSapitialAttention
from LA.CAM import CAM_Module
import torch
import torch.nn as nn
from thop import profile
from thop import clever_format

class LA(nn.Module):
    def __init__(self,in_channel):
        super(LA, self).__init__()


        self.msa = MulSapitialAttention(in_channel)
        # self.cam = CAM_Module(in_channel)
        self.mhsa = MulHeadSelfAttention(in_channel//2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel//2, in_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_channel // 2, affine=True),
            nn.SiLU(inplace=False)
        )

        pass
    def forward(self,x):

        out = self.msa(x)
        out = self.conv(out)
        out = self.mhsa(out)
        return out
        pass
    pass
if __name__ == '__main__':
    inputs = torch.randn(2, 64, 100, 100)
    self_attention = LA(64)
    out = self_attention(inputs)

    flops, params = profile(self_attention, inputs=(inputs,))
    flops, params = clever_format([flops,params],"%.3f")
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))
    print(out.shape)
    pass
