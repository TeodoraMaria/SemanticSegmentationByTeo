from PIL import Image
import os
import cv2
import numpy as np


def get_max_size():
    current_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    current_directory = os.getcwd()
    max_width = 0
    max_height = 0

    with open(current_directory + '/trainval.txt') as f:
        for line in f:
            line = line.rstrip('\n')
            im = Image.open(current_directory+'/train_images/'+line)
            width, height = im.size
            if width > max_width:
                max_width = width
            if height > max_height:
                max_height = height

    with open(current_directory + '/test.txt') as f:
        for line in f:
            line = line.rstrip('\n')
            im = Image.open(current_directory + '/test_images/' + line)
            width, height = im.size
            if width > max_width:
                max_width = width
            if height > max_height:
                max_height = height

    return max_width, max_height


def my_transform_image(width, height, img, isLabel=False):
    # BLACK = [0, 0, 0]
    # img1 = cv2.imread(img)
    # img_height, img_width, _ = img1.shape
    # top = int((height - img_height) / 2)
    # bottom = (height - img_height) - top
    # left = int((width - img_width) / 2)
    # right = (width - img_width) - left
    # constant = cv2.copyMakeBorder(img1, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    # constant = cv2.cvtColor(constant, cv2.COLOR_BGR2RGB)
    # pil_im = Image.fromarray(constant)

    old_im = Image.open(img)
    old_size = old_im.size

    new_size = (width, height)
    new_im = Image.new("RGB", new_size)  ## luckily, this is already black!
    if isLabel:
        new_im = Image.new("P", new_size)
    new_im.paste(old_im, (int((new_size[0] - old_size[0]) / 2),
                          int((new_size[1] - old_size[1]) / 2)))

    return new_im
