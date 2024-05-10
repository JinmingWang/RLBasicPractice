# 在这里定义神经网络模型的一些层和基础

import torch
import torch.nn as nn
import torch.nn.functional as func


def loadModel(model: nn.Module, model_path: str) -> None:
    """
    加载保存在本地的模型参数
    :param model: 需要加载的模型
    :param model_path: 保存的模型参数的路径
    :return:
    """
    model.load_state_dict(torch.load(model_path))


def saveModel(model: nn.Module, model_path: str) -> None:
    """
    保存模型参数到本地
    :param model: 需要保存的模型（包含经过训练的参数）
    :param model_path: 保存的模型参数的路径
    :return:
    """
    torch.save(model.state_dict(), model_path)


def copyModel(src: nn.Module, dst: nn.Module) -> None:
    """
    将src模型的参数复制到dst模型中
    :param src: 原模型
    :param dst: 目标模型
    :return:
    """
    dst.load_state_dict(src.state_dict())
    dst.eval()


class ConvBnReLU(nn.Sequential):
    """
    卷积神经网络里非常常用的结构：卷积层 + 批归一化层 + ReLU激活层
    这个算是固定的三件套了
    这个类继承了nn.Sequential，这是一个可以将多层组合在一起的容器模块，在nn.Sequential中的模块会像链表一样，
    其中的所有层依次地运行，前一层的输出作为后一层的输入
    """
    def __init__(self, in_c, out_c, k, s=1, p=0, d=1, g=1):
        """
        :param in_c: 输入通道数
        :param out_c: 输出通道数
        :param k: 卷积核大小
        :param s: 步长
        :param p: padding
        :param d: dilation
        :param g: groups
        """
        super(ConvBnReLU, self).__init__(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, dilation=d, groups=g, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )



