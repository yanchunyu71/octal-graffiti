import numpy
from image_augmentation import random_transform
from image_augmentation import random_warp

random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
}


def get_training_data(images, batch_size):
    indices = numpy.random.randint(len(images), size=batch_size)#返回一个随机整型数，范围从低（包括）到高（不包括
    for i, index in enumerate(indices):
        image = images[index]
        image = random_transform(image, **random_transform_args)#进行图片转换，图片增强
        warped_img, target_img = random_warp(image)#每一张图片都转换

        if i == 0:#两个新的空数组
            warped_images = numpy.empty((batch_size,) + warped_img.shape, warped_img.dtype)
            target_images = numpy.empty((batch_size,) + target_img.shape, warped_img.dtype)

        warped_images[i] = warped_img
        target_images[i] = target_img

    return warped_images, target_images
