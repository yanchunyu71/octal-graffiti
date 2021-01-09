import torch
import torch.nn as nn
from torchvision.models import vgg


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
                                                            #（一个批次数据个数、每张图片色彩通道数、图片高度、图片宽度）
        # input [b, 3, 150, 150] output [b, 512, 4, 4]  4维分别是（batch_size、channel、height、weight）,因为只截取了150*150大小
        self.base = vgg.vgg16_bn(True).features#卷积层是4维张量
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)#全连接层是二维张量，第一个参数是输入size，第二个参数是输出size，在转换过程中使用线性变化公式
        self.fc2 = nn.Linear(1024, 512 * 2 * 2)

    def forward(self, x):
        x = self.base(x)#先进入vgg卷积层                                                [b,512,4,4]
        b, c, h, w = x.size()#取出4维数                                
        x = x.view(b, c * h * w)# 对参数实现扁平化，转化为2维，将色彩通道数，高度，宽度融合[b,8192]
        x = self.fc1(x)# 全连接                                                          [b,1024]
        x = self.fc2(x)# 全连接                                                          [b,8192]
        x = x.view(b, c, h, w) #再转化为4维                                              [b,512,4,4]
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(512, 1024, 3, padding=2)#参数：输入通道数，输出通道数，卷积核深度，（卷积核移动步长），padding=2图像增加的边界层数
        self.conv2 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 3, 5, padding=2)
        self.lrelu = nn.LeakyReLU(0.1)# 激活函数，若输入的值为负，则将其斜率设为0.1
        self.ps = nn.PixelShuffle(2)# 将一个H × W的低分辨率输入图像（Low Resolution），通过Sub-pixel操作将其变为rH x rW的高分辨率图像，扩大倍率为2

    def forward(self, x):
        x = self.conv1(x)# [
        x = self.lrelu(x)# 
        x = self.ps(x)   # 

        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.ps(x)

        x = self.conv3(x)
        x = self.lrelu(x)
        x = self.ps(x)

        x = self.conv4(x)
        x = self.lrelu(x)
        x = self.ps(x)

        x = self.conv5(x)

        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder()# 一个编码器
        self.decoder_A = Decoder()# 解码器A
        self.decoder_B = Decoder()# 解码器B

    def forward(self, x, select="A"):
        if select == "A":
            x = self.encoder(x)
            out = self.decoder_A(x)
        else:
            x = self.encoder(x)
            out = self.decoder_B(x)
        return out

#测试经过网络的输入输出是否相同
if __name__ == "__main__":
    input = torch.rand((8, 3, 64, 64))#返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义 4维
    e = Encoder()
    output = e(input)
    print(output.size())

    input = torch.rand((8, 512, 2, 2))
    d = Decoder()
    output = d(input)
    print(output.size())
