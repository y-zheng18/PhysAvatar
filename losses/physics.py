import torch
import torch.nn as nn
from .layers import *


def collision_penalty(va, vb, nb, eps=1e-3, return_average=True, return_distance=False):
    closest_vertices = NearestNeighbour()(va, vb).long()
    vb = vb[closest_vertices]
    nb = nb[closest_vertices]

    distance = torch.sum(-nb * (va - vb), dim=-1)
    if return_distance:
        return distance

    interpenetration = torch.maximum(eps - distance, torch.tensor(0.0, device=va.device))

    if return_average:
        return torch.sum(interpenetration) / va.shape[0]

    return torch.sum(interpenetration ** 3)
