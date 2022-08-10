"""python
    PascalVOCDataset具体实现过程
"""
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    # 初始化相关变量
    # 读取images和objects标注信息
    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()  # 保证输入为纯大写字母，便于匹配{'TRAIN', 'TEST'}

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    # 循环读取image及对应objects
    # 对读取的image及objects进行tranform操作（数据增广）
    # 返回PIL格式图像，标注框，标注框对应的类别索引，对应的difficult标志(True or False)
    def __getitem__(self, i):
        # Read image
        # *需要注意，在pytorch中，图像的读取要使用Image.open()读取成PIL格式，不能使用opencv
        # *由于Image.open()读取的图片是四通道的(RGBA)，因此需要.convert('RGB')转换为RGB通道
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        # 如果self.keep_difficult为False,即不保留difficult标志为True的目标
        # 那么这里将对应的目标删去
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        # 对读取的图片应用transform
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    # 获取图片的总数，用于计算batch数
    def __len__(self):
        return len(self.images)

    # 我们知道，我们输入到网络中训练的数据通常是一个batch一起输入，而通过__getitem__我们只读取了一张图片及其objects信息
    # 如何将读取的一张张图片及其object信息整合成batch的形式呢？
    # collate_fn就是做这个事情，
    # 对于一个batch的images，collate_fn通过torch.stack()将其整合成4维tensor，对应的objects信息分别用一个list存储
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        # (3,224,224) -> (N,3,224,224)
        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 224, 224), 3 lists of N tensors each

