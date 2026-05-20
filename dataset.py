# 文件: dataset.py (最终正确版本)

import cv2
import numpy
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''

    def __init__(self, dataset, file_root='data/', transform=None):
        """
        dataset: 'train', 'val', or 'test'
        file_root: root of data_path, e.g. ./data/
        """
        # [MODIFIED] Store the dataset type ('train', 'val', etc.) to use it in __getitem__
        self.dataset_type = dataset

        self.file_list = open(file_root + '/list/' + dataset + '.txt').read().splitlines()
        self.pre_images = [file_root + '/A/' + x for x in self.file_list]
        self.post_images = [file_root + '/B/' + x for x in self.file_list]
        self.gts = [file_root + '/label/' + x for x in self.file_list]
        self.transform = transform

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_image_name = self.pre_images[idx]
        post_image_name = self.post_images[idx]
        label_name = self.gts[idx]

        pre_image = cv2.imread(pre_image_name)
        post_image = cv2.imread(post_image_name)
        label = cv2.imread(label_name, 0)

        img = numpy.concatenate((pre_image, post_image), axis=2)

        if self.transform:
            return self.transform(img, label)

        return img, label

    def get_img_info(self, idx):
        img = cv2.imread(self.pre_images[idx])
        return {"height": img.shape[0], "width": img.shape[1]}