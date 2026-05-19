import numpy
import numpy as np
import torch
import random
import cv2
import torchvision.transforms.functional as TF
from PIL import Image
import torchvision.transforms as T
from PIL import Image, ImageEnhance

class PrepareForTensorWithFlag(object):
    """
    一个用于验证/测试流程的转换。
    它不对图像做任何操作，仅仅在参数列表中添加一个值为0.0的'swapped'标志，
    以满足后续ToTensorWithFlag对输入格式的要求。
    """
    def __call__(self, image, label, pre_mask, post_mask):
        swapped = 0.0  # 对于验证/测试集，交换标志永远是0
        return [image, label, pre_mask, post_mask, swapped]

class ColorJitter(object):
    """
    对双时相图像进行同步的颜色抖动 (最终解决方案)。
    该版本使用稳定可靠的 PIL.ImageEnhance 模块，彻底绕开 torchvision.transforms.functional 的问题。
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, label, pre_mask, post_mask):
        # 假设输入的image是 (H, W, 6) 的numpy数组
        img1 = image[:, :, :3]
        img2 = image[:, :, 3:]

        img1_pil = Image.fromarray(img1.astype(np.uint8))
        img2_pil = Image.fromarray(img2.astype(np.uint8))

        # 决定应用顺序
        transforms_order = ['brightness', 'contrast', 'saturation', 'hue']
        random.shuffle(transforms_order)

        for transform_name in transforms_order:
            # 生成随机因子
            if transform_name == 'brightness':
                factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
                enhancer1 = ImageEnhance.Brightness(img1_pil)
                img1_pil = enhancer1.enhance(factor)
                enhancer2 = ImageEnhance.Brightness(img2_pil)
                img2_pil = enhancer2.enhance(factor)

            elif transform_name == 'contrast':
                factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
                enhancer1 = ImageEnhance.Contrast(img1_pil)
                img1_pil = enhancer1.enhance(factor)
                enhancer2 = ImageEnhance.Contrast(img2_pil)
                img2_pil = enhancer2.enhance(factor)

            elif transform_name == 'saturation':
                factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
                enhancer1 = ImageEnhance.Color(img1_pil)
                img1_pil = enhancer1.enhance(factor)
                enhancer2 = ImageEnhance.Color(img2_pil)
                img2_pil = enhancer2.enhance(factor)

            elif transform_name == 'hue':
                # [核心] PIL/Pillow 没有直接的 hue 调整，我们需要手动实现
                # 这也是 torchvision 内部必须做的事情，但我们可以用一种安全的方式来做

                # 确保 hue 在安全范围内
                safe_hue = min(self.hue, 0.5)
                hue_factor = random.uniform(-safe_hue, safe_hue)

                # 如果 hue_factor 为 0，则无需调整
                if abs(hue_factor) > 1e-6:
                    # 将 RGB 转换为 HSV
                    img1_hsv = np.array(img1_pil.convert('HSV'))
                    img2_hsv = np.array(img2_pil.convert('HSV'))

                    # --- 安全的色调调整 ---
                    # H 通道在 HSV 中是 0-255 的 uint8
                    # 我们的 hue_factor 是 [-0.5, 0.5]
                    # 调整量是 hue_factor * 255
                    hue_shift = hue_factor * 255.0

                    # 先转换为 float 进行计算，避免溢出
                    h1 = img1_hsv[:, :, 0].astype(np.float32)
                    h2 = img2_hsv[:, :, 0].astype(np.float32)

                    h1 = (h1 + hue_shift) % 256
                    h2 = (h2 + hue_shift) % 256

                    # 转回 uint8
                    img1_hsv[:, :, 0] = h1.astype(np.uint8)
                    img2_hsv[:, :, 0] = h2.astype(np.uint8)

                    # 将 HSV 转换回 RGB
                    img1_pil = Image.fromarray(img1_hsv, 'HSV').convert('RGB')
                    img2_pil = Image.fromarray(img2_hsv, 'HSV').convert('RGB')

        img1_aug = np.array(img1_pil)
        img2_aug = np.array(img2_pil)

        image_aug = np.concatenate([img1_aug, img2_aug], axis=2)

        return [image_aug, label, pre_mask, post_mask]


class RandomGaussianBlur(object):
    """
    以一定概率对双时相图像进行同步的高斯模糊。
    """

    def __init__(self, kernel_size=3, prob=0.5):
        self.kernel_size = kernel_size
        self.prob = prob

    def __call__(self, image, label, pre_mask, post_mask):
        if random.random() < self.prob:
            # 分离图像
            img1 = image[:, :, :3]
            img2 = image[:, :, 3:]

            # 对两个图像应用相同的高斯模糊
            # 注意：cv2.GaussianBlur 要求核尺寸为奇数
            ksize = self.kernel_size if self.kernel_size % 2 != 0 else self.kernel_size + 1
            img1_blurred = cv2.GaussianBlur(img1, (ksize, ksize), 0)
            img2_blurred = cv2.GaussianBlur(img2, (ksize, ksize), 0)

            # 合并图像
            image_blurred = np.concatenate([img1_blurred, img2_blurred], axis=2)

            return [image_blurred, label, pre_mask, post_mask]

        return [image, label, pre_mask, post_mask]

class Scale(object):
    """
    Resize the given image to a fixed scale
    """

    def __init__(self, wi, he):
        '''
        :param wi: width after resizing
        :param he: height after reszing
        '''
        self.w = wi
        self.h = he

    # modified from torchvision to add support for max size

    def __call__(self, img, label,pre_mask,post_mask):
        '''
        :param img: RGB image
        :param label: semantic label image
        :return: resized images
        '''
        # bilinear interpolation for RGB image
        img = cv2.resize(img, (self.w, self.h))
        # nearest neighbour interpolation for label image
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        pre_mask = cv2.resize(pre_mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        post_mask = cv2.resize(post_mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return [img, label,pre_mask,post_mask]


class Resize(object):
    def __init__(self, min_size, max_size, strict=False):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.strict = strict

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        if not self.strict:
            size = random.choice(self.min_size)
            max_size = self.max_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)
        else:
            if w < h:
                return (self.max_size, self.min_size[0])
            else:
                return (self.min_size[0], self.max_size)

    def __call__(self, image, label,pre_mask,post_mask):
        size = self.get_size(image.shape[:2])
        # print("origin", image.shape)
        image = cv2.resize(image, size)
        # print("resized", image.shape)
        # print('*'*20)
        # I confirm that the output size is right, not reversed
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        pre_mask = cv2.resize(pre_mask, size, interpolation=cv2.INTER_NEAREST)
        post_mask = cv2.resize(post_mask, size, interpolation=cv2.INTER_NEAREST)
        return (image, label,pre_mask,post_mask)


class RandomCropResize(object):
    """
    Randomly crop and resize the given image with a probability of 0.5
    """

    def __init__(self, crop_area):
        '''
        :param crop_area: area to be cropped (this is the max value and we select between 0 and crop area
        '''
        self.cw = crop_area
        self.ch = crop_area

    def __call__(self, img, label,pre_mask,post_mask):
        if random.random() < 0.5:
            h, w = img.shape[:2]
            x1 = random.randint(0, self.ch)
            y1 = random.randint(0, self.cw)

            img_crop = img[y1:h - y1, x1:w - x1]
            label_crop = label[y1:h - y1, x1:w - x1]
            pre_mask_crop = pre_mask[y1:h - y1, x1:w - x1]
            post_mask_crop = post_mask[y1:h - y1, x1:w - x1]

            img_crop = cv2.resize(img_crop, (w, h))
            label_crop = cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST)
            pre_mask_crop = cv2.resize(pre_mask_crop, (w, h), interpolation=cv2.INTER_NEAREST)
            post_mask_crop = cv2.resize(post_mask_crop, (w, h), interpolation=cv2.INTER_NEAREST)

            return img_crop, label_crop,pre_mask_crop,post_mask_crop
        else:
            return [img, label,pre_mask,post_mask]


class RandomFlip(object):
    """
    Randomly flip the given Image with a probability of 0.5
    """

    def __call__(self, image, label,pre_mask,post_mask):
        if random.random() < 0.5:
                image = cv2.flip(image, 0)  # horizontal flip
                label = cv2.flip(label, 0)  # horizontal flip
                pre_mask = cv2.flip(pre_mask, 0)  # horizontal flip
                post_mask = cv2.flip(post_mask, 0)  # horizontal flip
        if random.random() < 0.5:
                image = cv2.flip(image, 1)  # veritcal flip
                label = cv2.flip(label, 1)  # veritcal flip
                pre_mask = cv2.flip(pre_mask, 1)  # veritcal flip
                post_mask = cv2.flip(post_mask, 1)  # veritcal flip
        return [image, label,pre_mask,post_mask]


class RandomExchange(object):
    """
    Randomly flip the given Image with a probability of 0.5
    AND return a flag indicating if the swap happened.
    """

    def __call__(self, image, label, pre_mask, post_mask):
        was_swapped = False  # 默认未交换
        if random.random() < 0.5:
            pre_img = image[:, :, 0:3]
            post_img = image[:, :, 3:6]
            image = numpy.concatenate((post_img, pre_img), axis=2)
            was_swapped = True  # 标记已交换

        # [核心修改] 返回一个额外的 'was_swapped' 标志
        return image, label, pre_mask, post_mask, was_swapped


class Normalize(object):
    # ... __init__ aunchanged ...
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, label, pre_mask, post_mask, *args): # 使用 *args 接收额外参数
        image = image.astype(np.float32)
        image = image / 255
        label = np.ceil(label / 255)
        pre_mask = np.ceil(pre_mask / 255)
        post_mask = np.ceil(post_mask / 255)
        for i in range(6):
            image[:, :, i] -= self.mean[i]
        for i in range(6):
            image[:, :, i] /= self.std[i]

        return (image, label, pre_mask, post_mask) + args # 将额外参数原样传回


class GaussianNoise(object):
    def __init__(self, std=0.05):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''
        self.std = std

    def __call__(self, image, label,pre_mask,post_mask):
        noise = np.random.normal(loc=0, scale=self.std, size=image.shape)
        image = image + noise.astype(np.float32)
        return [image, label,pre_mask,post_mask]


class ToTensor(object):
    # ... __init__ aunchanged ...
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, image, label, pre_mask, post_mask, *args):  # 使用 *args 接收额外参数
        # ... aunchanged logic for image and label processing ...
        image = image[:, :, ::-1].copy()
        image = image.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image)
        label_tensor = torch.LongTensor(np.array(label, dtype=np.int32)).unsqueeze(dim=0)
        pre_tensor = torch.LongTensor(np.array(pre_mask, dtype=np.int32)).unsqueeze(dim=0)
        post_tensor = torch.LongTensor(np.array(post_mask, dtype=np.int32)).unsqueeze(dim=0)

        # 将原始 python bool/int 转换为 tensor
        processed_args = [torch.tensor(arg) for arg in args]

        return [image_tensor, label_tensor, pre_tensor, post_tensor] + processed_args


class Compose(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
