"""Self-ensemble inference helpers."""

import torch


def _forward_transformed(x, s, hflip, vflip, rotate, model):
    """Apply one transform, run the model, invert the transform, and return the prediction."""
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
    """Average predictions over flip and rotation transforms for one input batch."""
    outputs = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rotate in [False, True]:
                outputs.append(_forward_transformed(x, s, hflip, vflip, rotate, model))
    return torch.stack(outputs).mean(dim=0)
