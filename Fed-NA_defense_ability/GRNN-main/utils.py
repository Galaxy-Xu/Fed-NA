# -*-coding:utf-8-*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
import numpy as np
import PIL.Image as Image
from WassersteinDistance.wd import wasserstein_distance
from skimage.metrics import structural_similarity as ssim

def flatten_gradients(dy_dx):
    flatten_dy_dx = None
    for layer_g in dy_dx:
        if flatten_dy_dx is None:
            flatten_dy_dx = torch.flatten(layer_g)
        else:
            flatten_dy_dx = torch.cat((flatten_dy_dx, torch.flatten(layer_g)))
    return flatten_dy_dx

def GC_flatten_gradients(tensor):
    return tensor.cpu().numpy().flatten()

device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 从路径中读取数据并将图片变为指定大小，返回值为数据集（dst）和本数据对应的种类数目(num_class)
def gen_dataset(dataset, data_path, shape_img):
    class Dataset_from_Image(Dataset):
        def __init__(self, imgs, labs, transform=None):
            self.imgs = imgs
            self.labs = labs
            self.transform = transform
            del imgs, labs

        def __len__(self):
            return self.labs.shape[0]

        def __getitem__(self, idx):
            lab = self.labs[idx]
            img = Image.open(self.imgs[idx])
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = self.transform(img)
            return img, lab

    # def face_dataset(path, num_classes):
    #     images_all = []
    #     index_all = []
    #     folders = os.listdir(path)
    #     for foldidx, fold in enumerate(folders):
    #         if foldidx+1==num_classes: break
    #         if os.path.isdir(os.path.join(path, fold)):
    #             files = os.listdir(os.path.join(path, fold))
    #             for f in files:
    #                 if len(f) > 4:
    #                     images_all.append(os.path.join(path, fold, f))
    #                     index_all.append(foldidx)
    #     transform = transforms.Compose([transforms.Resize(shape_img),
    #                              transforms.ToTensor()])
    #     dst = Dataset_from_Image(images_all, np.asarray(index_all, dtype=int), transform=transform)
    #     return dst
    if dataset == 'mnist':
        num_classes = 10
        # 按照tt的格式对图像进行处理
        tt = transforms.Compose([transforms.Resize(shape_img),   # 图像大小变为 shape * shape
                                 transforms.Grayscale(num_output_channels=3),  # 图像通道数变为3
                                 transforms.ToTensor()
                                 ])
        dst = datasets.MNIST(os.path.join(data_path, 'mnist/'), download=True, transform=tt)
    elif dataset == 'cifar100':
        num_classes = 100
        tt = transforms.Compose([transforms.Resize(shape_img),
                                 transforms.ToTensor()])
        dst = datasets.CIFAR100(os.path.join(data_path, 'cifar100/'), download=True, transform=tt)
    elif dataset == 'fashion-mnist':
        num_classes = 10
        tt = transforms.Compose([transforms.Resize(shape_img),   # 图像大小变为 shape * shape
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dst = datasets.FashionMNIST(os.path.join(data_path, 'fashion-mnist/'), download=True, transform=tt)

    # elif dataset == 'lfw':
    #     num_classes = 5749
    #     dst = face_dataset(os.path.join(data_path, 'lfw/'), shape_img)
    # elif dataset == 'VGGFace':
    #     num_classes = 2622
    #     dst = face_dataset(os.path.join(data_path, 'VGGFace/vgg_face_dataset/'), num_classes)
    else:
        exit('unknown dataset')
    return dst, num_classes

def weights_init(m):
    try:
        if hasattr(m, 'weight'):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    @staticmethod
    def _tensor_size(t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def loss_f(loss_name, flatten_fake_g, flatten_true_g, device):
    if loss_name == 'l2':
        grad_diff = ((flatten_fake_g - flatten_true_g) ** 2).sum()
        # grad_diff = torch.sqrt(((flatten_fake_g - flatten_true_g) ** 2).sum())
    elif loss_name == 'wd':
        grad_diff = wasserstein_distance(flatten_fake_g.view(1, -1), flatten_true_g.view(1, -1),
                                         device=f'cuda:{device}')
    else:
        raise Exception('Wrong loss name.')
    return grad_diff

# def SSIM(dummy_data, ground_truth):
#     # Assuming for now that the matching dummy_data and gt are given
#     dummy_data = torch.flatten(dummy_data)
#     ground_truth = torch.flatten(ground_truth)
#
#     dummy_data_mean = torch.mean(dummy_data.double()).item()
#     ground_truth_mean = torch.mean(ground_truth.double()).item()
#     dummy_mean_square = dummy_data_mean**2
#     gt_mean_square = ground_truth_mean**2
#
#     dummy_data_var = torch.var(dummy_data, unbiased=False).item()
#     ground_truth_var = torch.var(ground_truth, unbiased=False).item()
#     cov = covar(dummy_data, ground_truth)
#
#     c1 = 0.0001
#     c2 = 0.0009
#
#     term1 = (2 * dummy_data_mean * ground_truth_mean) + c1
#     term2 = (2 * cov) + c2
#     term3 = dummy_mean_square + gt_mean_square + c1
#     term4 = dummy_data_var + ground_truth_var + c2
#
#     return (term1 * term2) / (term3 * term4)
#
# def covar(a, b):
#     a_bar = torch.mean(a).item()
#     b_bar = torch.mean(b).item()
#     cov = 0
#     for i in range(len(a)):
#         cov += (a[i].item() - a_bar) * (b[i].item() - b_bar)
#     return cov / (len(a) - 1)

def SSIM(dummy_data, ground_truth):
    # Flatten the tensors
    dummy_data = dummy_data.view(-1)
    ground_truth = ground_truth.view(-1)

    dummy_data_mean = dummy_data.mean().item()
    ground_truth_mean = ground_truth.mean().item()
    dummy_mean_square = dummy_data_mean**2
    gt_mean_square = ground_truth_mean**2

    dummy_data_var = dummy_data.var(unbiased=False).item()
    ground_truth_var = ground_truth.var(unbiased=False).item()
    cov = torch.dot(dummy_data - dummy_data_mean, ground_truth - ground_truth_mean) / (len(dummy_data) - 1)

    c1 = 0.0001
    c2 = 0.0009

    term1 = (2 * dummy_data_mean * ground_truth_mean) + c1
    term2 = (2 * cov) + c2
    term3 = dummy_mean_square + gt_mean_square + c1
    term4 = dummy_data_var + ground_truth_var + c2

    return (term1 * term2) / (term3 * term4)

