from __future__ import print_function, division
import os

import numpy as np
import scipy.io
import torch.utils.data as data
from PIL import Image
from mypath import Path

from torchvision import transforms
from dataloaders import custom_transforms as tr

class SBDSegmentation(data.Dataset):
    NUM_CLASSES = 21

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('sbd'),
                 split='train', mode='train'
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self.mode = mode
        self.split = split
        # self._dataset_dir = os.path.join(self._base_dir, 'dataset')
        # self._image_dir = os.path.join(self._dataset_dir, 'img')
        # self._cat_dir = os.path.join(self._dataset_dir, 'cls')
        _split_f = os.path.join(self._base_dir, split+'.txt')
        assert os.path.isfile(_split_f), "%s not found" % _split_f

        self.args = args

        # Get list of all images from the split and check that the files exist
        self.im_ids = []
        self.images = []
        self.categories = []

        with open(_split_f, "r") as f:
            lines = f.read().splitlines()
        for line in lines:
            _image, _categ = line.strip("\n").split(' ')

            _image = os.path.join(self._base_dir, _image)
            assert os.path.isfile(_image), '%s not found' % _image
            if split in ['train_augvoc', 'val_voc']:
                _categ = os.path.join(self._base_dir, _categ.lstrip('/'))
            assert os.path.isfile(_image)
            assert os.path.isfile(_categ)
            self.im_ids.append(line)
            self.images.append(_image)
            self.categories.append(_categ)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images: {:d}'.format(len(self.images)))


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}
        if self.mode == 'train':
            return self.transform(sample), os.path.basename(self.images[index])
        else:
            return self.transform_eval(sample), os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target

    def transform(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomRotate(15),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_eval(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def denorm(self, image):
        means = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [C*H*W]"
            assert image.size(0) == 3, "Expected RGB image [3*H*W]"
            for t,m,s in zip(image,means, std):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            assert image.size(1) == 3, "Expected RGB image [3*h*w]"
            for t,m,s in zip((0,1,2), means, std):
                image[:, t, :,:].mul_(s).add_(m)
        return image



    def __str__(self):
        return 'SBDSegmentation(split=' + str(self.split) + ')'


class SBDSegmentation_test(data.Dataset):
    NUM_CLASSES = 21

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('sbd'),
                 split='test'
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self.split = split
        # self._dataset_dir = os.path.join(self._base_dir, 'dataset')
        # self._image_dir = os.path.join(self._dataset_dir, 'img')
        # self._cat_dir = os.path.join(self._dataset_dir, 'cls')
        _split_f = os.path.join(self._base_dir, split+'.txt')
        assert os.path.isfile(_split_f), "%s not found" % _split_f

        self.args = args

        # Get list of all images from the split and check that the files exist
        self.im_ids = []
        self.images = []

        with open(_split_f, "r") as f:
            lines = f.read().splitlines()
        for line in lines:
            _image = line.strip("\n").split(' ')[0]

            _image = os.path.join(self._base_dir, _image)
            assert os.path.isfile(_image), '%s not found' % _image
            assert os.path.isfile(_image)
            self.im_ids.append(line)
            self.images.append(_image)
        # Display stats
        print('Number of images: {:d}'.format(len(self.images)))


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}
        return self.transform_eval(sample), os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        # _target = Image.open(self.categories[index])
        w,h =_img.size
        _target = Image.fromarray(np.zeros((h,w)))
        return _img, _target

    def transform(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_eval(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def denorm(self, image):
        means = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [C*H*W]"
            assert image.size(0) == 3, "Expected RGB image [3*H*W]"
            for t,m,s in zip(image,means, std):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            assert image.size(1) == 3, "Expected RGB image [3*h*w]"
            for t,m,s in zip((0,1,2), means, std):
                image[:, t, :,:].mul_(s).add_(m)
        return image



    def __str__(self):
        return 'SBDSegmentation(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    sbd_train = SBDSegmentation(args, split='train')
    dataloader = DataLoader(sbd_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)