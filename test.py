from PIL import Image
from torch import nn
from model import VGG16

import torchvision
import numpy as np
import cv2.cv2 as cv2
import torch
import argparse


parser = argparse.ArgumentParser(description='VGG16 Testing')
parser.add_argument("--weight_dir", default='./weights/A_Early_stop_params.pth', help="参数路径")
parser.add_argument("--test_dir", default='./dataset/test/img/Image_21.jpg', help="测试图片路径")

args = parser.parse_args()

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = VGG16(mode="test")
net = net.to(device=device)

# 加载模型参数
net.load_state_dict(torch.load(args.weight_dir))

# 进行推理
def run():
    net.eval()
    image = cv2.imread(args.test_dir, cv2.IMREAD_COLOR)
    # image = np.float32(image)
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor()])
    image = trans(image).unsqueeze(0).to(device)
    result = torch.argmax(net(image).ravel())
    # result = net(image)
    return result


if __name__ == '__main__':
    result = run()
    print(result)
