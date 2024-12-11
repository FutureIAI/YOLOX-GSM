import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from thop import clever_format


class CAM_Module(nn.Module):
    def __init__(self,in_dim):
        super(CAM_Module,self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim,in_dim//2,kernel_size=4,stride=2,padding=1),
            nn.InstanceNorm2d(in_dim // 2, affine=True),
            nn.SiLU(inplace=False)
        )

    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        out = self.conv(out)
        return out
    pass

if __name__ == '__main__':
    inputs = torch.randn(1, 32, 100, 100)

    # print(type(inputs))
    # temp = inputs.unsqueeze(dim=1)
    # print(temp.shape)
    # print(inputs)
    # print(torch.max(inputs,dim=1))

    # self_attention = ResBlock_CBAM(32,64,stride=2,downsampling=True)
    self_attention = CAM_Module(32)
    out = self_attention(inputs)
    print(out.shape)

    flops, params = profile(self_attention, inputs=(inputs,))
    flops, params = clever_format([flops, params], "%.3f")
    print(out.shape)
    print('flops:{}'.format(flops))
    print('params:{}'.format(params))

    pass
