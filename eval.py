import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.metrics import Evaluator
import scipy.misc
from PIL import Image, ImagePalette


def denorm(image):
    means = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if image.dim() == 3:
        assert image.dim() == 3, "Expected image [C*H*W]"
        assert image.size(0) == 3, "Expected RGB image [3*H*W]"
        for t, m, s in zip(image, means, std):
            t.mul_(s).add_(m)
    elif image.dim() == 4:
        assert image.size(1) == 3, "Expected RGB image [3*h*w]"
        for t, m, s in zip((0, 1, 2), means, std):
            image[:, t, :, :].mul_(s).add_(m)
    return image

def colormap(N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'uint8'
    cmap = []
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap.append((r, g, b))

    return cmap

def _get_voc_pallete(num_cls):
    n = num_cls
    pallete = [0]*(n*3)
    for j in range(0,n):
            lab = j
            pallete[j*3+0] = 0
            pallete[j*3+1] = 0
            pallete[j*3+2] = 0
            i = 0
            while (lab > 0):
                    pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                    pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                    pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                    i = i + 1
                    lab >>= 3
    return pallete

def save_img(root_path, img_name, pred, img, pallete):
    def mask2rgb(mask):
        # print(mask.shape)
        im = Image.fromarray(np.uint8(mask)).convert("P")
        im.putpalette(pallete)
        mask_rgb = np.array(im.convert("RGB"), dtype=np.float)
        return mask_rgb / 255.

    def mask_overlay(mask, image, alpha=0.3):
        mask_rgb = mask2rgb(mask)
        image = np.transpose(image, (1, 2, 0))
        return alpha*image + (1-alpha)*mask_rgb

    filepath = os.path.join(root_path, 'mask', img_name+'.png')
    scipy.misc.imsave(filepath, pred.astype(np.uint8))

    overlay = mask_overlay(pred, img)
    filepath = os.path.join(root_path, 'vis', img_name+'.png')
    overlay255 = np.round(overlay*255).astype(np.uint8)
    scipy.misc.imsave(filepath, overlay255)


class Evaler(object):
    def __init__(self, args, palette):
        self.args = args
        self.palette = palette
        self.root_path = args.save_dir
        if not os.path.exists(self.root_path):
            os.mkdir(self.root_path)
        if not os.path.exists(os.path.join(self.root_path, 'vis')):
            os.mkdir(os.path.join(self.root_path, 'vis'))
        if not os.path.exists(os.path.join(self.root_path, 'mask')):
            os.mkdir(os.path.join(self.root_path, 'mask'))

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)


        self.model = model
        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    def validation(self):
        self.model.eval()
        tbar = tqdm(self.val_loader, desc='\r')
        for i, (sample, img_name) in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # print(pred.shape)
            target = target.cpu().numpy()
            self.evaluator.add_batch(target, pred)
            pred = pred[0, :, :]
            orig_img = denorm(image[0, :, :, :]).cpu().numpy()
            # print(img_name)
            save_img(self.root_path, img_name=img_name[0], pred=pred, img=orig_img, pallete=self.palette)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Validation:')
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval'],)
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--split', type=str, default='train_augvoc',
                        choices=['train_augvoc', 'val_voc', 'train_gen_bsl'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    # training hyper params
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')

    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='put the path to resuming file if needed')

    args = parser.parse_args()
    args.split = 'val_voc'
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    print(args)
    cmap = colormap()
    palette = ImagePalette.ImagePalette()
    for rgb in cmap:
        palette.getcolor(rgb)
    torch.manual_seed(args.seed)
    trainer = Evaler(args, palette)
    trainer.validation()

if __name__ == "__main__":
   main()
