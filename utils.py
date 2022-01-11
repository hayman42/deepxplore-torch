import torch
import torchvision.transforms as transforms
import torchvision.models as models

import itertools
import random
import os
import numpy as np
from PIL import Image


def get_img(dir_path):
    while True:
        orig_img = random.choice(os.listdir(dir_path))
        if orig_img == ".git":
            continue
        img = Image.open(dir_path + orig_img)
        img = img.resize((224, 224), Image.LANCZOS)
        return transforms.ToTensor()(img).reshape(-1, 3, 224, 224), orig_img


def to_image(img_tensor):
    img = img_tensor.squeeze(0).detach()
    img = img.transpose(0, 2).transpose(0, 1).numpy()
    img[..., 0] *= 255
    img[..., 1] *= 255
    img[..., 2] *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def output_table(model):
    d = {}
    neurons = set()
    for name in dict(model.named_parameters()).keys():
        if "weight" in name:
            neurons.add(name[:-7])

    def set_table(name):
        def hook(model, i, o):
            d[name] = o
        return hook

    for name, layer in model.named_modules():
        if "fc" in name:
            continue
        if name in neurons:
            layer.register_forward_hook(set_table(name))
    return d


def scale(o):
    return (o - o.min()) / (o.max() - o.min())


def init_coverage(coverage_table, output_table):
    for key in output_table.keys():
        out = output_table[key][0]
        coverage_table[key] = torch.zeros(out.shape[0], dtype=torch.bool)


def update_coverage(coverage_table, output_table, threshold):
    for key in output_table.keys():
        scaled = scale(output_table[key][0])
        if len(scaled.shape) > 1:
            scaled = scaled.mean(dim=list(range(1,len(scaled.shape))))
        coverage_table[key] |= (scaled > threshold)


def neuron_coverage(coverage_table):
    activated = sum(map(lambda key: coverage_table[key].sum(), coverage_table.keys()))
    neurons = sum(map(lambda key: len(coverage_table[key]), coverage_table.keys()))
    return activated / neurons


def neuron_to_cover(coverage_table):
    to_cover = []
    for layer in coverage_table.keys():
        to_cover.extend(itertools.product([layer], torch.where(coverage_table[layer] == False)[0]))
    return random.choice(to_cover)


def compute_obj1(c, out, lambda1):
    loss = sum(o[c] for o in out[1:])
    loss -= out[0][c] * lambda1
    return loss


def compute_obj2(coverage_tables, output_tables):
    neurons_to_cover = [neuron_to_cover(cov) for cov in coverage_tables]
    loss = 0
    for (layer, index), o in zip(neurons_to_cover, output_tables):
        loss += o[layer][0][index, ...].mean()
    return loss


def constraint_light(grad):
    return 1e4 * grad.mean() * torch.ones_like(grad)


def constraint_black(grads, rect_shape=(10, 10)):
    start_point = (
        random.randint(0, grads.shape[2] - rect_shape[0]), random.randint(0, grads.shape[3] - rect_shape[1]))
    new_grads = torch.zeros_like(grads)
    patch = grads[..., start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if torch.mean(patch) < 0:
        new_grads[..., start_point[0]:start_point[0] + rect_shape[0],
                  start_point[1]:start_point[1] + rect_shape[1]] = -torch.ones_like(patch)
    return new_grads


def constraint_occl(grads, start_point, rect_shape):
    new_grads = torch.zeros_like(grads)
    new_grads[..., start_point[0]:start_point[0] + rect_shape[0],
              start_point[1]:start_point[1] + rect_shape[1]] = grads[..., start_point[0]:start_point[0] + rect_shape[0],
                                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads
