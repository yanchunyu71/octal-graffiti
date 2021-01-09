import cv2
import numpy

from umeyama import umeyama

random_transform_args = {
    'rotation_range': 10,#旋转
    'zoom_range': 0.05,#缩放变换
    'shift_range': 0.05,#平移变换
    'random_flip': 0.4,#翻转
}

#随机转换
def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    h, w = image.shape[0:2]
    rotation = numpy.random.uniform(-rotation_range, rotation_range)#生出参数范围内随机个符合均分布的浮点数-旋转范围，
    scale = numpy.random.uniform(1 - zoom_range, 1 + zoom_range)#生成让图片在长或宽的方向进行放大,可以理解为某方向的resize
    tx = numpy.random.uniform(-shift_range, shift_range) * w
    ty = numpy.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)#获得仿射变化矩阵
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)#进行仿射变化，根据变换矩阵对源矩阵进行变换
    if numpy.random.random() < random_flip:
        result = result[:, ::-1]
    return result

#随机变形
# get pair of random warped images from aligened face image
def random_warp(image):
    assert image.shape == (256, 256, 3)
    range_ = numpy.linspace(128 - 80, 128 + 80, 5)#在指定的间隔内返回均匀间隔的数字
    mapx = numpy.broadcast_to(range_, (5, 5))#将数组广播到新形状
    mapy = mapx.T#数组转置

    mapx = mapx + numpy.random.normal(size=(5, 5), scale=5)#random.normal生成高斯分布的概率密度随机数
    mapy = mapy + numpy.random.normal(size=(5, 5), scale=5)

    interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype('float32')
    interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype('float32')

    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)#重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程
                    #其实就是一个坐标到另一个坐标的映射remap参数（输入图像， ， ，插值方式）
    src_points = numpy.stack([mapx.ravel(), mapy.ravel()], axis=-1)#讲数组进行堆叠，前是原数组，后是要在axis维上堆叠
    dst_points = numpy.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)#返回多维结构，常见的如2D图形，3D图形,表示在0:65间均匀取数，取16个
    mat = umeyama(src_points, dst_points, True)[0:2]# Umeyama：返回一个变换矩阵，通过缩放，旋转平移对齐到标准位置

    target_image = cv2.warpAffine(image, mat, (64, 64))#map是获得的变换矩阵，warpAffine是将源矩阵根据map进行变换
                    #warpAffine参数（输入图像，变换矩阵，输出图像的大小）
    return warped_image, target_image
