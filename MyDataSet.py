import torch.utils.data as data
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch

from Helpers.ColorMapLabels import color_map
from Helpers.ImageSizeDecider import get_max_size, my_transform_image

transform_image = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])


class MyDataSet(data.Dataset):
    def __init__(self, filename, img_dir, transform=None):
        self.imgs = []
        self.img_dir = img_dir
        self.transform = transform
        current_directory = os.getcwd()
        self.label_dir = os.path.join(current_directory, r'label_images')
        self.dimensions = [500,500]
        self.color_map = color_map()

        with open(filename) as f:
            for line in f:
                self.imgs.append(line.rstrip())
        print(len(self.imgs))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        im = my_transform_image(self.dimensions[0], self.dimensions[1],
                                               "{}/{}".format(self.img_dir, img))
        img = img.replace("jpg", "png")
        lab = my_transform_image(self.dimensions[0], self.dimensions[1], "{}/{}".format(self.label_dir, img), True)

        lab = transform_image(lab)

        # lab = np.array(lab)
        im = transform_image(im)
        # im = np.array(im)
        lab=lab*255
        lab[lab == 255] = 0
        im=im*255

        #lab = torch.from_numpy(lab)
        # im = torch.from_numpy(im)
        lab = lab.view(lab.size(1), lab.size(2))


        return im, lab.long()
