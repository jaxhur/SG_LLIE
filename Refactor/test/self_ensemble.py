"""测试时使用的 self-ensemble 工具。

self-ensemble 会对输入做翻转/旋转，分别推理后再反变换并平均，
通常能略微提升测试结果，但会明显增加推理时间和显存占用。
"""

import torch


def _forward_transformed(x, s, hflip, vflip, rotate, model):
    """执行一次带变换的前向推理。

    输入:
        x: 输入图像张量。
        s: 结构先验张量。
        hflip/vflip/rotate: 是否执行水平翻转、垂直翻转、旋转。
        model: SG_LLIE 模型。
    输出:
        反变换后的增强结果。
    """
    if hflip:
        x = torch.flip(x, dims=(-2,))
        s = torch.flip(s, dims=(-2,))
    if vflip:
        x = torch.flip(x, dims=(-1,))
        s = torch.flip(s, dims=(-1,))
    if rotate:
        x = torch.rot90(x, dims=(-2, -1))
        s = torch.rot90(s, dims=(-2, -1))
    output = model(x, s)[0]
    if rotate:
        output = torch.rot90(output, dims=(-2, -1), k=3)
    if vflip:
        output = torch.flip(output, dims=(-1,))
    if hflip:
        output = torch.flip(output, dims=(-2,))
    return output


def self_ensemble(x, s, model):
    """对 8 种翻转/旋转组合的预测结果求平均。

    输入:
        x: 输入图像张量。
        s: 结构先验张量。
        model: SG_LLIE 模型。
    输出:
        平均后的增强图像张量。
    """
    outputs = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rotate in [False, True]:
                outputs.append(_forward_transformed(x, s, hflip, vflip, rotate, model))
    return torch.stack(outputs).mean(dim=0)
