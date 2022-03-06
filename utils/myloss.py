from torch import nn

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyLoss(nn.Module):
    # 该函数等价于nn.CrossEntropyLoss
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, pred, label, eps=1e-3):
        # label的shape:tensor([pic1_class, pic2_class])
        batch_size = pred.shape[0]  # 获取预测值的batch维度
        new_pred = pred.reshape(batch_size, -1)  # 将输出预测张量转化为[batch, class_num]
        expand_target = torch.zeros(new_pred.shape, device=device)  # 建立一个和new_pred一样大的张量
        for i in range(batch_size):
            expand_target[i, int(label[i])] = 1  # 给对应图片类别的位置赋值为1
        softmax_pred = torch.softmax(new_pred, dim=1)  # 把输出预测进行softmax,这里进行softmax了！
        return torch.sum(-torch.log(softmax_pred + eps) * expand_target) / batch_size  # 计算总损失

    # def __call__(self, *args, **kwargs):
    #     return self.forward(*args)


if __name__ == '__main__':
    a = torch.tensor([[0.0791, -0.2797, 0.5169, -0.1229, 0.4389],
                      [-0.1366, 0.0622, 0.1356, 0.2859, 0.5595]], device=device)
    b = torch.tensor([[0], [1]], device=device)
    loss = MyLoss()
    myloss = loss(a, b)
    print(myloss)
