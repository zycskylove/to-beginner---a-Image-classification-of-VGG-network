from PIL import Image

import numpy as np
import torchvision


class DataAugment(object):
    def __init__(self):
        super(DataAugment, self).__init__()

    def detect_resize(self, img, label, size: tuple):
        trans = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size=size),
            torchvision.transforms.ToTensor()]
        )
        image = trans(img)
        return image, label

    def __call__(self, *args, **kwargs):  # args:tuple    kwargs:dict
        return self.detect_resize(*args)
