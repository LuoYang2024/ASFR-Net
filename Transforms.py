import numpy as np
import torch
import random
import cv2
from PIL import Image, ImageEnhance


class PrepareForTensorWithFlag(object):
    def __call__(self, image, label):
        swapped = 0.0
        return [image, label, swapped]


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, label):
        img1 = image[:, :, :3]
        img2 = image[:, :, 3:]
        img1_pil = Image.fromarray(img1.astype(np.uint8))
        img2_pil = Image.fromarray(img2.astype(np.uint8))

        transforms_order = ['brightness', 'contrast', 'saturation', 'hue']
        random.shuffle(transforms_order)

        for transform_name in transforms_order:
            if transform_name == 'brightness':
                factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
                img1_pil = ImageEnhance.Brightness(img1_pil).enhance(factor)
                img2_pil = ImageEnhance.Brightness(img2_pil).enhance(factor)
            elif transform_name == 'contrast':
                factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
                img1_pil = ImageEnhance.Contrast(img1_pil).enhance(factor)
                img2_pil = ImageEnhance.Contrast(img2_pil).enhance(factor)
            elif transform_name == 'saturation':
                factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
                img1_pil = ImageEnhance.Color(img1_pil).enhance(factor)
                img2_pil = ImageEnhance.Color(img2_pil).enhance(factor)
            elif transform_name == 'hue':
                safe_hue = min(self.hue, 0.5)
                hue_factor = random.uniform(-safe_hue, safe_hue)
                if abs(hue_factor) > 1e-6:
                    img1_hsv = np.array(img1_pil.convert('HSV'))
                    img2_hsv = np.array(img2_pil.convert('HSV'))
                    hue_shift = hue_factor * 255.0
                    img1_hsv[:, :, 0] = ((img1_hsv[:, :, 0].astype(np.float32) + hue_shift) % 256).astype(np.uint8)
                    img2_hsv[:, :, 0] = ((img2_hsv[:, :, 0].astype(np.float32) + hue_shift) % 256).astype(np.uint8)
                    img1_pil = Image.fromarray(img1_hsv, 'HSV').convert('RGB')
                    img2_pil = Image.fromarray(img2_hsv, 'HSV').convert('RGB')

        image_aug = np.concatenate([np.array(img1_pil), np.array(img2_pil)], axis=2)
        return [image_aug, label]


class RandomGaussianBlur(object):
    def __init__(self, kernel_size=3, prob=0.5):
        self.kernel_size = kernel_size if kernel_size % 2 != 0 else kernel_size + 1
        self.prob = prob

    def __call__(self, image, label):
        if random.random() < self.prob:
            img1 = cv2.GaussianBlur(image[:, :, :3], (self.kernel_size, self.kernel_size), 0)
            img2 = cv2.GaussianBlur(image[:, :, 3:], (self.kernel_size, self.kernel_size), 0)
            return [np.concatenate([img1, img2], axis=2), label]
        return [image, label]


class Scale(object):
    def __init__(self, wi, he):
        self.w, self.h = wi, he

    def __call__(self, img, label):
        img = cv2.resize(img, (self.w, self.h))
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        return [img, label]


class Resize(object):
    def __init__(self, min_size, max_size, strict=False):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.strict = strict

    def get_size(self, image_size):
        w, h = image_size
        if not self.strict:
            size = random.choice(self.min_size)
            if self.max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > self.max_size:
                    size = int(round(self.max_size * min_original_size / max_original_size))
            if w < h:
                return (int(size * h / w), size)
            else:
                return (size, int(size * w / h))
        else:
            return (self.max_size, self.min_size[0]) if w < h else (self.min_size[0], self.max_size)

    def __call__(self, image, label):
        size = self.get_size(image.shape[:2])
        image = cv2.resize(image, size[::-1])  # cv2.resize 使用 (width, height)
        label = cv2.resize(label, size[::-1], interpolation=cv2.INTER_NEAREST)
        return (image, label)


class RandomCropResize(object):
    def __init__(self, crop_area):
        self.cw = crop_area
        self.ch = crop_area

    def __call__(self, img, label):
        if random.random() < 0.5:
            h, w = img.shape[:2]
            x1 = random.randint(0, self.ch)
            y1 = random.randint(0, self.cw)
            img_crop = img[y1:h - y1, x1:w - x1]
            label_crop = label[y1:h - y1, x1:w - x1]
            img_crop = cv2.resize(img_crop, (w, h))
            label_crop = cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST)
            return img_crop, label_crop
        else:
            return [img, label]


class RandomFlip(object):
    def __call__(self, image, label):
        if random.random() < 0.5:
            image, label = cv2.flip(image, 0), cv2.flip(label, 0)
        if random.random() < 0.5:
            image, label = cv2.flip(image, 1), cv2.flip(label, 1)
        return [image, label]


class RandomExchange(object):
    def __call__(self, image, label):
        was_swapped = False
        if random.random() < 0.5:
            image = np.concatenate((image[:, :, 3:6], image[:, :, 0:3]), axis=2)
            was_swapped = True
        return image, label, was_swapped


class Normalize(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, image, label, *args):
        image = image.astype(np.float32) / 255.0
        label = np.ceil(label / 255.0)
        for i in range(6):
            image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]
        return (image, label) + args


class GaussianNoise(object):
    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, image, label):
        noise = np.random.normal(loc=0, scale=self.std, size=image.shape)
        image = (image + noise.astype(np.float32)).clip(0, 255)
        return [image, label]


class ToTensor(object):
    def __call__(self, image, label, *args):
        # RGB -> BGR 并转置为 (C, H, W)
        image = image[:, :, ::-1].copy().transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image)
        label_tensor = torch.LongTensor(np.array(label, dtype=np.int32)).unsqueeze(dim=0)
        processed_args = [torch.tensor(arg) for arg in args]
        return [image_tensor, label_tensor] + processed_args


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args