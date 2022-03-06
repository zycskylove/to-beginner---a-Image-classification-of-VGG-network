from math import ceil
from torch.utils.data import DataLoader
from utils.myloss import MyLoss
from model import VGG16
from dataset.read_yolo_dataset import ReadYOLO
from Augmentation.data_augment import DataAugment

import torch
import argparse
import time

parser = argparse.ArgumentParser(description='VGG16 Training')
parser.add_argument("--lr", default=0.001, help="learning rate of model")  # 0.1~0.0001
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
parser.add_argument('--epochs', default=20, type=int, help='epochs')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取数据集
data_augment = DataAugment()
dataset = ReadYOLO(phase="train", trans=data_augment, device=device)
picture_num = len(dataset)  # 获取图片的总数量

# 模型实例化
net = VGG16()
net.train()
net = net.to(device=device)

# 迭代器和损失函数优化器实例化
optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss = MyLoss()


# 创建图片数据迭代器
def colle(batch):
    # 假设一个batch_size=2:那么batch的shape就是((picture1_tensor, picture1_label), (picture2_tensor, picture2_label))
    imgs, targets = list(zip(*batch))  # 这里通过解压batch把多张图片的picture_tensor放在一起，picture_label放在一起
    imgs = torch.cat(imgs, dim=0)
    targets = torch.cat(targets, dim=0)
    return imgs, targets


data = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=colle)


def train():
    # 设置epoch
    epochs = args.epochs
    batch_count = 0  # 对batch进行计数
    for epoch in range(epochs):
        for batch, (imgs, targets) in enumerate(data):
            start = time.time()
            pred = net(imgs)  # (batch_size, 3,224,224)
            Loss = loss(pred, targets)
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            # 训练完一个batch需要的时间
            batch_time = time.time() - start
            # 剩下的训练完需要的迭代次数
            total_num = epochs * ceil(picture_num / args.batch_size) - batch_count
            # 剩下的训练完需要的时间，返回的是秒
            rest_time = total_num * batch_time
            # 转化为h/m/s
            hour = int(rest_time / 60 // 60)
            minute = int(rest_time // 60)
            second = int((rest_time / 60 - minute) * 60)
            batch_count += 1
            print("epoch:{0}/{1} ".format(epoch+1, epochs),
                  "  ,loss: ", float(Loss),
                  "  ,每个batch所需时间: ", batch_time,
                  "  ,剩余批次: ", total_num,
                  "  ,eta: {0}小时{1}分钟{2}秒".format(hour, minute, second))
            if Loss <= 1e-3:
                torch.save(net.state_dict(), "./weights/An_Early_stop_params.pth".format(epoch + 1))
                return print("训练结束！!")
        # 每个epoch保存一次参数
        torch.save(net.state_dict(), "./weights/VGG16_epoch{}_params.pth".format(epoch+1))
    print("训练结束！!")


if __name__ == '__main__':
    train()
