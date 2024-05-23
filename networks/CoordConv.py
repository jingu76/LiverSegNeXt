import torch.nn as nn
import torch
import torch.nn.functional as F

class CoordConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.norm = nn.BatchNorm3d(in_channels)
        # 改变通道数
        self.conv = nn.Conv3d(in_channels+3, out_channels, kernel_size=1, stride=1, padding=0)
        self.activation = nn.LeakyReLU(0.2)


    def forward(self,x,coord):
        # ct标准化
        x = self.norm(x)
        # 合坐标信息合并
        out = torch.cat([x, coord], dim=1)
        # 改变通道
        out = self.conv(out)
        # 激活
        out = self.activation(out)
        return out
		
		

# 1.将ct标准化
# 2.和坐标信息合并
# 3.改变通道
# 4.激活	

if __name__ == '__main__':
    x = torch.randn(4, 1, 128, 128, 32).cuda()
    coord=torch.randn(4, 3, 128, 128, 32).cuda()
    
    conv=CoordConvBlock(1,1).cuda()
    x=conv(x,coord)
    print(x.shape)
    