import torch

def flatten_gradients(dy_dx):
    flatten_dy_dx = None
    for layer_g in dy_dx:
        if flatten_dy_dx is None:
            flatten_dy_dx = torch.flatten(layer_g)
        else:
            flatten_dy_dx = torch.cat((flatten_dy_dx, torch.flatten(layer_g)))
    return flatten_dy_dx