import torch.nn as nn
import torch
import torch.nn.functional as F
from networks.CoordConv import CoordConvBlock
from monai.networks.blocks import UnetrBasicBlock


# 黄色的模块
class conv_block(nn.Module):
    # scale表示扩张倍数，k_size表示卷积核的大小
    def __init__(self, c_in, scale, k_size):
        super().__init__()
        # 3D卷积,groups=c_in 表示每个输入通道都有一个卷积核，这使其成为深度卷积
        # padding=(k_size - 1) // 2 用于保持输入和输出的大小相同,可以根据公式推导
        self.dw_conv1 = nn.Conv3d(
            c_in,
            c_in,
            kernel_size=k_size,
            groups=c_in,
            stride=1,
            padding=1,
            bias=False,
            dilation=1,
        )
        self.dw_conv2 = nn.Conv3d(
            c_in,
            c_in,
            kernel_size=k_size,
            groups=c_in,
            stride=1,
            padding=3,
            bias=False,
            dilation=3,
        )
        self.dw_conv3 = nn.Conv3d(
            c_in,
            c_in,
            kernel_size=k_size,
            groups=c_in,
            stride=1,
            padding=5,
            bias=False,
            dilation=5,
        )

        # 扩大通道数
        self.expansion = nn.Conv3d(c_in, c_in * scale, kernel_size=(1, 1, 1), stride=1)
        # 激活函数
        self.act = nn.GELU()
        # 压缩通道数
        self.compress = nn.Conv3d(c_in * scale, c_in, kernel_size=(1, 1, 1), stride=1)
        # 将通道分成32组进行归一化
        self.norm = nn.GroupNorm(32, c_in)

    def forward(self, x):
        identity = x

        out1 = self.norm(self.dw_conv1(x))
        out2 = self.norm(self.dw_conv2(x))
        out3 = self.norm(self.dw_conv3(x))
        out = self.norm(out1 + out2 + out3)

        out = self.act(self.expansion(out))
        out = self.compress(out)
        out = torch.add(out, identity)
        return out


# down,蓝色的模块
class down_block(nn.Module):
    def __init__(self, c_in, scale, k_size):
        super().__init__()
        self.dw_conv = nn.Conv3d(
            c_in,
            c_in,
            kernel_size=k_size,
            groups=c_in,
            stride=2,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.norm = nn.GroupNorm(32, c_in)
        self.expansion = nn.Conv3d(c_in, scale * c_in, kernel_size=(1, 1, 1), stride=1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(
            scale * c_in, 2 * c_in, kernel_size=(1, 1, 1), stride=1
        )
        self.shortcut = nn.Conv3d(c_in, 2 * c_in, kernel_size=(1, 1, 1), stride=2)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.norm(self.dw_conv(x))
        out = self.act(self.expansion(out))
        out = self.compress(out)
        out = torch.add(out, shortcut)
        return out


# 绿色的模块，up
class up_block(nn.Module):
    def __init__(self, c_in, scale, k_size):
        super().__init__()
        self.dw_conv = nn.ConvTranspose3d(
            c_in,
            c_in,
            kernel_size=k_size,
            stride=2,
            padding=(k_size - 1) // 2,
            output_padding=1,
            groups=c_in,
            bias=False,
        )
        self.norm = nn.GroupNorm(32, c_in)
        self.expansion = nn.Conv3d(c_in, c_in * scale, kernel_size=(1, 1, 1), stride=1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(
            c_in * scale, c_in // 2, kernel_size=(1, 1, 1), stride=1
        )
        self.shortcut = nn.ConvTranspose3d(
            c_in, c_in // 2, kernel_size=(1, 1, 1), output_padding=1, stride=2
        )

    def forward(self, x1, x2):
        short = self.shortcut(x1)
        x1 = self.norm(self.dw_conv(x1))
        x1 = self.act(self.expansion(x1))
        x1 = self.compress(x1)
        x1 = torch.add(x1, short)
        out = torch.add(x1, x2)
        return out


class MedNeXt(nn.Module):
    def __init__(self, in_channel, base_c, k_size, num_block, scale, num_class):
        super().__init__()
        self.stem = nn.Conv3d(in_channel, base_c, kernel_size=(1, 1, 1), stride=1)
        self.layer1 = self._make_layer(base_c, num_block[0], scale[0], k_size)
        self.down1 = down_block(base_c, scale[1], k_size)
        self.layer2 = self._make_layer(base_c * 2, num_block[1], scale[1], k_size)
        self.down2 = down_block(base_c * 2, scale[2], k_size)
        self.layer3 = self._make_layer(base_c * 4, num_block[2], scale[2], k_size)
        self.down3 = down_block(base_c * 4, scale[3], k_size)
        self.layer4 = self._make_layer(base_c * 8, num_block[3], scale[3], k_size)
        self.down4 = down_block(base_c * 8, scale[4], k_size)
        self.layer5 = self._make_layer(base_c * 16, num_block[4], scale[4], k_size)

        self.up1 = up_block(base_c * 16, scale[4], k_size)
        self.layer6 = self._make_layer(base_c * 8, num_block[5], scale[5], k_size)
        self.up2 = up_block(base_c * 8, scale[5], k_size)
        self.layer7 = self._make_layer(base_c * 4, num_block[6], scale[6], k_size)
        self.up3 = up_block(base_c * 4, scale[6], k_size)
        self.layer8 = self._make_layer(base_c * 2, num_block[7], scale[7], k_size)
        self.up4 = up_block(base_c * 2, scale[7], k_size)
        self.layer9 = self._make_layer(base_c, num_block[8], scale[8], k_size)

        self.out = nn.Conv3d(base_c, num_class, kernel_size=(1, 1, 1), stride=1)

        # 坐标卷积
        # self.coordconv0=CoordConvBlock(16*32,16*32)
        # self.coordconv1=CoordConvBlock(8*32,8*32)
        # self.coordconv2=CoordConvBlock(4*32,4*32)
        # self.coordconv3=CoordConvBlock(2*32,2*32)
        self.coordconv4 = CoordConvBlock(1 * 32, 1 * 32)

        # 两阶段转换通道的卷积层
        # self.conv1 = nn.Conv3d(33, 32, kernel_size=1, stride=1, padding=0)
        # self.conv2 = nn.Conv3d(65, 64, kernel_size=1, stride=1, padding=0)
        # self.conv3 = nn.Conv3d(129,128, kernel_size=1, stride=1, padding=0)
        # self.conv4 = nn.Conv3d(257,256, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv3d(513, 512, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, c_in, n_conv, ratio, k_size):
        layers = []
        for _ in range(n_conv):
            layers.append(conv_block(c_in, ratio, k_size))
        return nn.Sequential(*layers)

    def forward(self, input):
        # 分开CT和坐标信息
        x = input[:, 0, :, :, :]
        x = torch.unsqueeze(x, dim=1)
        # 准备好坐标信息
        coord1 = input[:, 1:4, :, :, :]
        coord2 = F.avg_pool3d(coord1, kernel_size=2, stride=2)
        coord3 = F.avg_pool3d(coord2, kernel_size=2, stride=2)
        coord4 = F.avg_pool3d(coord3, kernel_size=2, stride=2)
        coord5 = F.avg_pool3d(coord4, kernel_size=2, stride=2)
        # 准备好第一阶段预测的结果
        predict = input[:, 4, :, :, :]
        predict1 = torch.unsqueeze(predict, dim=1)
        predict2 = F.avg_pool3d(predict1, kernel_size=2, stride=2)
        predict3 = F.avg_pool3d(predict2, kernel_size=2, stride=2)
        predict4 = F.avg_pool3d(predict3, kernel_size=2, stride=2)
        predict5 = F.avg_pool3d(predict4, kernel_size=2, stride=2)

        # MedNeXt模型
        x = self.stem(x)
        # 在第1层添加
        x = self.coordconv4(x, coord1)

        out1 = self.layer1(x)
        d1 = self.down1(out1)
        out2 = self.layer2(d1)
        d2 = self.down2(out2)
        out3 = self.layer3(d2)
        d3 = self.down3(out3)
        out4 = self.layer4(d3)
        d4 = self.down4(out4)
        out_5 = self.layer5(d4)

        # 两阶段,添加标签
        out_5 = torch.cat([out_5, predict5], dim=1)
        # 改变通道
        out_5 = self.conv5(out_5)

        up1 = self.up1(out_5, out4)
        out6 = self.layer6(up1)
        up2 = self.up2(out6, out3)
        out7 = self.layer7(up2)
        up3 = self.up3(out7, out2)

        out8 = self.layer8(up3)
        up4 = self.up4(out8, out1)

        out9 = self.layer9(up4)
        out = self.out(out9)
        return out


# 选择论文中的B设置
def get_mednet(in_channels, out_channels, kernel_size):
    # 论文中的Bi
    num_block = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    # 论文中的Ri
    scale = [2, 3, 4, 4, 4, 4, 4, 3, 2]
    # 修改输入频道是1,输出频道是4
    net = MedNeXt(
        in_channel=1,
        base_c=32,
        k_size=kernel_size,
        num_block=num_block,
        scale=scale,
        num_class=out_channels,
    )
    return net


# debug
if __name__ == "__main__":
    net = get_mednet(1, 2, 3)
    x = torch.rand(1, 4, 128, 128, 32)

    out = net(x)
    print(out.shape)
