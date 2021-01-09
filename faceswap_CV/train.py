import argparse
import os

import cv2
import numpy as np
import torch

import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

# from models import Autoencoder, toTensor, var_to_np
from models import toTensor, var_to_np
from my_model import Autoencoder
from util import get_image_paths, load_images, stack_images
from training_data import get_training_data

parser = argparse.ArgumentParser(description='faceswap_CV')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cuda = True

if args.cuda is True:
    print('===> Using GPU to train')
    device = torch.device('cuda:0')
    cudnn.benchmark = True
else:
    print('===> Using CPU to train')

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print('===> Loaing datasets')
images_A = get_image_paths("./face_A")#获取图片路径
images_B = get_image_paths("./face_B")
images_A = load_images(images_A) / 255.0#将图片加载到数组中，并将NumPy数组规范化到一定范围[0,255]内
images_B = load_images(images_B) / 255.0
images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))#将图像A集加上两者图像集的平均差值(RGB三通道差值)，axis=0,1,2表示在这三个轴上取平均
                                                                         #来使两个输入图像图像的分布尽可以相近，这样我们的损失函数曲线下降会更快些
model = Autoencoder().to(device)#定义模型

print('===> Try resume from checkpoint')
if os.path.isdir('checkpoint'):
    try:
        checkpoint = torch.load('./checkpoint/autoencoder.t7')
        model.load_state_dict(checkpoint['state'])
        start_epoch = checkpoint['epoch']
        print('===> Load last checkpoint data')
    except FileNotFoundError:
        print('Can\'t found autoencoder.t7')
else:
    start_epoch = 0
    print('===> Start from scratch')

#定义损失函数和优化器
criterion = nn.L1Loss()#损失函数使用L1
optimizer_1 = optim.Adam([{'params': model.encoder.parameters()},   #A的优化器（待优化参数的参数组的dict[],lr学习率5*10^-5,
                          {'params': model.decoder_A.parameters()}]         # betas用于计算梯度以及梯度平方的运行平均值的系数)
                         , lr=5e-5, betas=(0.5, 0.999))
optimizer_2 = optim.Adam([{'params': model.encoder.parameters()},    #B的优化器
                          {'params': model.decoder_B.parameters()}]
                         , lr=5e-5, betas=(0.5, 0.999))


if __name__ == "__main__":

    print('Start training, press \'q\' to stop')

    for epoch in range(start_epoch, args.epochs):
        batch_size = args.batch_size

        warped_A, target_A = get_training_data(images_A, batch_size)#warped是数据增强之后的图片，需要有一个和他对应的目标图片target（因为二者不能完全一样）
        warped_B, target_B = get_training_data(images_B, batch_size)

        warped_A, target_A = toTensor(warped_A), toTensor(target_A)#转化为tensor张量
        warped_B, target_B = toTensor(warped_B), toTensor(target_B)

        if args.cuda:
            warped_A = warped_A.to(device).float()# 将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行
            target_A = target_A.to(device).float()# .float()将该tensor投射为float类型 
            warped_B = warped_B.to(device).float()# variable是floattensor的封装
            target_B = target_B.to(device).float()# 似乎这里就已将把tensor转换为Variable了？

        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        warped_A = model(warped_A, 'A')#使用A解码器训练A
        warped_B = model(warped_B, 'B')#使用B解码器训练B

        loss1 = criterion(warped_A, target_A)#取预测值和目标值的绝对误差的平均数
        loss2 = criterion(warped_B, target_B)
        loss = loss1.item() + loss2.item()#loss合并
        loss1.backward()#反向传播，依次计算并存储神经网络中间变量和参数的梯度
        loss2.backward()
        optimizer_1.step()#更新梯度
        optimizer_2.step()
        print('epoch: {}, lossA:{}, lossB:{}'.format(epoch, loss1.item(), loss2.item()))

        if epoch % args.log_interval == 0:

            test_A_ = target_A[0:14]# b = a[n:m]表示列表切片，复制列表a[n]到a[m-1]的内容到新的列表对象b[]
            test_B_ = target_B[0:14]
            test_A = var_to_np(target_A[0:14])#Variable转化为numpy
            test_B = var_to_np(target_B[0:14])
            print('===> Saving models...')
            state = {
                'state': model.state_dict(),
                'epoch': epoch
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/autoencoder.t7')

        # 使用训练图片测试
        figure_A = np.stack([
            test_A,
            var_to_np(model(test_A_, 'A')),#先输入模型，然后转换为numpy
            var_to_np(model(test_A_, 'B')),#换脸
        ], axis=1)
        figure_B = np.stack([
            test_B,
            var_to_np(model(test_B_, 'B')),
            var_to_np(model(test_B_, 'A')),#换脸
        ], axis=1)

        figure = np.concatenate([figure_A, figure_B], axis=0)#能够一次完成多个数组的拼接。其中a1,a2,...是数组类型的参数
        figure = figure.transpose((0, 1, 3, 4, 2))
        figure = figure.reshape((4, 7) + figure.shape[1:])
        figure = stack_images(figure)

        figure = np.clip(figure * 255, 0, 255).astype('uint8')

        # cv2.imshow("", figure)
        cv2.imwrite("./train_result.jpg", figure)
        key = cv2.waitKey(1)
        if key == ord('q'):
            exit()
