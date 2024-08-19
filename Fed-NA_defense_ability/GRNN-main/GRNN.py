import time, datetime
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import *
from Generator.model import Generator
from TFLogger.logger import TFLogger
from Backbone.lenet import LeNet
from Backbone.resnet import ResNet18
from Backbone.MLP import MLP
from RandSumZero import creat_noise_1, creat_noise_2
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device0 = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for GRNN training
    device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu") # for local training, if you have only one GPU, please set device1 to 0
    batchsize = 128
    save_img = True # whether save generated image and its relevant true image
    Iteration = 3000 # how many optimization steps on GRNN
    num_exp = 1 # experiment number
    g_in = 16
    plot_num = 50
    net_name = 'lenet' # global model
    net_name_set = ['lenet', 'res18','MLP']
    dataset = 'fashion-mnist'
    dataset_set = ['mnist', 'cifar100', 'lfw', 'VGGFace', 'ilsvrc','fashion-mnist']
    shape_img = (32, 32)
    root_path = os.path.abspath(os.curdir)
    print('root_path ---> '+ root_path)
    data_path = os.path.join(root_path, 'Data/')
    save_path = os.path.join(root_path, f"Results/GRNN-{net_name}-{dataset}-S{shape_img[0]}-B{str(batchsize).zfill(3)}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}/") # path to saving results
    print('>' * 10, save_path)
    save_img_path = os.path.join(save_path, 'saved_img/')
    dst, num_classes= gen_dataset(dataset, data_path, shape_img) # read local data
    tp = transforms.Compose([transforms.ToPILImage()])
    train_loader = iter(torch.utils.data.DataLoader(dst, batch_size=batchsize, shuffle=True,generator=torch.Generator(device='cuda')))
    criterion = nn.CrossEntropyLoss().cuda(device1)
    print(f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}: {save_path}')
    for idx_net in range(num_exp):
        train_tfLogger = TFLogger(f'{save_path}/tfrecoard-exp-{str(idx_net).zfill(2)}') # tensorboard record
        print(f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}: running {idx_net+1}|{num_exp} experiment')
        if net_name == 'lenet':
            net = LeNet(num_classes=num_classes)
        elif net_name == 'res18':
            net = ResNet18(num_classes=num_classes)
        elif net_name == 'MLP':
            net = MLP(num_classes=num_classes)

        net = net.cuda(device1)
        Gnet = Generator(num_classes, channel=3, shape_img=shape_img[0],
                         batchsize=batchsize, g_in=g_in).cuda(device0)
        net.apply(weights_init)
        Gnet.weight_init(mean=0.0, std=0.02)
        G_optimizer = torch.optim.RMSprop(Gnet.parameters(), lr=0.0001, momentum=0.99)
        tv_loss = TVLoss()
        gt_data,gt_label = next(train_loader)
        gt_data, gt_label = gt_data.cuda(device1), gt_label.cuda(device1) # assign to device1 to generate true graident
        pred = net(gt_data)
        y = criterion(pred, gt_label)
        dy_dx = torch.autograd.grad(y, net.parameters()) # obtain true gradient

#.............................................Soteria.........................................................
        # feature_fc1_graph = dy_dx
        #
        # deviation_f1_target = torch.zeros_like(dy_dx)
        # deviation_f1_x_norm = torch.zeros_like(dy_dx)
        # for f in range(deviation_f1_x_norm.size(1)):
        #     deviation_f1_target[:, f] = 1
        #     feature_fc1_graph.backward(deviation_f1_target, retain_graph=True)
        #     deviation_f1_x = gt_data.grad.data
        #     deviation_f1_x_norm[:, f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1) / (
        #                 feature_fc1_graph.data[:, f] + 0.1)
        #     net.zero_grad()
        #     gt_data.grad.data.zero_()
        #     deviation_f1_target[:, f] = 0
        # deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
        # thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), 70)
        # mask = np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)

#...............................................Outpost...........................................................
        #
        # var = []
        # for param in net.parameters():
        #     var.append(torch.var(param).cpu().detach().numpy())
        # var = [min(v, 1) for v in var]
        #
        # # Calculate empirical FIM
        # fim = []
        # flattened_fim = None
        # for i in range(len(dy_dx)):
        #     squared_grad = dy_dx[i].clone().pow(2).mean(0).cpu().numpy()
        #     fim.append(squared_grad)
        #     if flattened_fim is None:
        #         flattened_fim = squared_grad.flatten()
        #     else:
        #         flattened_fim = np.append(flattened_fim, squared_grad.flatten())
        #
        # #fim_thresh = np.percentile(flattened_fim, 100 - 40)
        # fim_thresh = np.percentile(flattened_fim, 60)
        #
        # num = 0
        #
        # for i in range(len(dy_dx)):
        #     # pruning
        #     grad_tensor = dy_dx[i].cpu().numpy()
        #     flattened_weights = np.abs(grad_tensor.flatten())
        #     thresh = np.percentile(flattened_weights, 80)
        #     grad_tensor = np.where(abs(grad_tensor) < thresh, 0, grad_tensor)
        #     # noise
        #     noise_base = torch.normal(0, var[num] * 0.8, dy_dx[i].shape)
        #
        #     #noise_mask = np.where(fim[i] < fim_thresh, 0, 1)
        #     noise_mask = np.where(fim[i] < fim_thresh, 0, 1)
        #     if noise_mask.sum() == 0:
        #         noise_mask[0] = 1
        #
        #     #gauss_noise = noise_base * noise_mask
        #     gauss_noise = (noise_base.cpu() * torch.FloatTensor(noise_mask)).cuda(device1)
        #     #dy_dx[i] = (torch.Tensor(grad_tensor) + gauss_noise).to(dtype=torch.float32).to(device1)
        #     dy_dx_list = list(dy_dx)
        #     dy_dx_list[i] = (torch.Tensor(grad_tensor) + gauss_noise).to(dtype=torch.float32).to(device1)
        #     dy_dx = tuple(dy_dx_list)
        #
        #     num += 1

#.................................................gradient compression..............................................................
        # dy_dx = list(dy_dx)
        #
        # for i in range(len(dy_dx)):
        #     grad_tensor = dy_dx[i].cpu().data.numpy()
        #     flattened_weights = GC_flatten_gradients(dy_dx[i])
        #
        #     # Generate the pruning threshold according to 'prune by percentage'.
        #     thresh = np.percentile(np.abs(flattened_weights), 70)
        #
        #     grad_tensor_pruned = np.where(np.abs(grad_tensor) < thresh, 0, grad_tensor)
        #
        #     # Move grad_tensor_pruned back to CUDA device1
        #     dy_dx[i] = torch.Tensor(grad_tensor_pruned).to(device1)
        #
        # # Convert dy_dx back to tuple
        # dy_dx = tuple(dy_dx)

#..................................................Fed-NA.............................................................
#..................................................Add differential privacy noise...........................................................
        # target_layer_num = 8
        #
        # flatten_before_num = 0
        # for i in range(target_layer_num):
        #     flatten_before_num += len(flatten_gradients(dy_dx[i]))
        #
        # target_layer_flat = flatten_gradients(dy_dx[target_layer_num])
        # noise_tensor = torch.from_numpy(np.random.laplace(loc=0, scale=0.1, size=target_layer_flat.shape)).to(device1)
        # length = len(target_layer_flat)
        #
        #
        # flatten_true_g = flatten_gradients(dy_dx)
        #
        # for i in range(length):
        #     flatten_true_g[flatten_before_num+i] += noise_tensor[i]
 #................................................Add zero-sum noise........................................................
        # noise_convolution_1 = creat_noise_1()
        # for i in range(450):
        #     flatten_true_g[i] += noise_convolution_1[9,i]
        #
        # noise_convolution_2 = creat_noise_2()
        # for j in [456,2856]:
        #     for k in range(2400):
        #         flatten_true_g[j] += noise_convolution_2[5,k]

#.....................................。。.....DPFL........................................................
        # flatten_true_g = flatten_gradients(dy_dx)
        # noise_tensor = torch.from_numpy(np.random.laplace(loc=0, scale=0.1, size=flatten_true_g.shape)).to(device1)
        # flatten_true_g += noise_tensor
#.............................................。fedavg.......................................................
        flatten_true_g = flatten_gradients(dy_dx)
        G_ran_in = torch.randn(batchsize, g_in).cuda(device0) # initialize GRNN input


        iter_bar = tqdm(range(Iteration),
                        total=Iteration,
                        desc=f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}',
                        ncols=180)
        history = []
        history_l = []
        for iters in iter_bar: # start  optimizing GRNN
            # Gout = [batchsize * channel * image_shape * image_shape]      Glabel = [batchsize * num_class]
            Gout, Glabel = Gnet(G_ran_in) # produce recovered data
            Gout, Glabel = Gout.cuda(device1), Glabel.cuda(device1) # assign to device1 as global model's input to generate fake gradient
            Gpred = net(Gout)
            Gloss = criterion(Gpred, Glabel)
            # Gloss = - torch.mean(torch.sum(Glabel * torch.log(torch.softmax(Gpred, 1)), dim=-1)) # cross-entropy loss
            G_dy_dx = torch.autograd.grad(Gloss, net.parameters(), create_graph=True) # obtain fake gradient
            flatten_fake_g = flatten_gradients(G_dy_dx).cuda(device1)
            grad_diff_l2 = loss_f('l2', flatten_fake_g, flatten_true_g, device1)
            grad_diff_wd = loss_f('wd', flatten_fake_g, flatten_true_g, device1)
            if net_name == 'lenet':
                tvloss = 1e-3 * tv_loss(Gout)
            elif net_name == 'res18':
                tvloss = 1e-6 * tv_loss(Gout)
            elif net_name == 'MLP':
                tvloss = 1e-6 * tv_loss(Gout)

            grad_diff = grad_diff_l2 + grad_diff_wd + tvloss # loss for GRNN
            G_optimizer.zero_grad()
            grad_diff.backward()
            G_optimizer.step()
            iter_bar.set_postfix(loss_l2 = np.round(grad_diff_l2.item(), 8),
                                 loss_wd=np.round(grad_diff_wd.item(), 8),
                                 loss_tv = np.round(tvloss.item(), 8),
                                 img_mses=round(torch.mean(abs(Gout-gt_data)).item(), 8),
                                 img_wd=round(wasserstein_distance(Gout.view(1,-1), gt_data.view(1,-1)).item(), 8),
                                 img_ssim = round(float(SSIM(Gout,gt_data)), 8))

            train_tfLogger.scalar_summary('g_l2', grad_diff_l2.item(), iters)
            train_tfLogger.scalar_summary('g_wd', grad_diff_wd.item(), iters)
            train_tfLogger.scalar_summary('g_tv', tvloss.item(), iters)
            train_tfLogger.scalar_summary('img_mses', torch.mean(abs(Gout-gt_data)).item(), iters)
            train_tfLogger.scalar_summary('img_wd', wasserstein_distance(Gout.view(1,-1), gt_data.view(1,-1)).item(), iters)
            train_tfLogger.scalar_summary('img_ssim', SSIM(Gout, gt_data),iters)
            train_tfLogger.scalar_summary('toal_loss', grad_diff.item(), iters)

            # Gout = [batchsize * channel * image_shape * image_shape]      Glabel = [batchsize * num_class]
            # 这个循环的次数等于 plot的数目
            if iters % int(Iteration / plot_num) == 0:
                history.append([tp(Gout[imidx].detach().cpu()) for imidx in range(batchsize)])
                history_l.append([Glabel.argmax(dim=1)[imidx].item() for imidx in range(batchsize)])
                # print(history_l[-1])
            torch.cuda.empty_cache()
            del Gloss, G_dy_dx, flatten_fake_g, grad_diff_l2, grad_diff_wd, grad_diff, tvloss


        # visualization
        for imidx in range(batchsize):
            plt.figure(figsize=(12, 8))
            plt.subplot(plot_num//10, 10, 1)
            plt.imshow(tp(gt_data[imidx].cpu()))
            for i in range(min(len(history), plot_num-1)):
                plt.subplot(plot_num//10, 10, i + 2)
                plt.imshow(history[i][imidx])
                plt.title('l=%d' % (history_l[i][imidx]))
                # plt.title('i=%d,l=%d' % (history_iters[i], history_l[i][imidx]))
                plt.axis('off')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if save_img:
                true_path = os.path.join(save_img_path, f'true_data/exp{str(idx_net).zfill(3)}/')
                fake_path = os.path.join(save_img_path, f'fake_data/exp{str(idx_net).zfill(3)}/')
                if not os.path.exists(true_path) or not os.path.exists(fake_path):
                    os.makedirs(true_path)
                    os.makedirs(fake_path)
                tp(gt_data[imidx].cpu()).save(os.path.join(true_path, f'{imidx}_{gt_label[imidx].item()}.png'))
                history[-1][imidx].save(os.path.join(fake_path, f'{imidx}_{Glabel.argmax(dim=1)[imidx].item()}.png'))
            plt.savefig(os.path.join(save_img_path,f'{idx_net}_{imidx}_{gt_label[imidx].item()}_{Glabel.argmax(dim=1)[imidx].item()}.png'))
            #plt.savefig(save_img_path + '/exp:%03d-imidx:%02d-tlabel:%d-Glabel:%d.png' % (idx_net, imidx, gt_label[imidx].item(), Glabel.argmax(dim=1)[imidx].item()))
            plt.close()

        del Glabel, Gout, flatten_true_g, G_ran_in, net, Gnet
        torch.cuda.empty_cache()
        history.clear()
        history_l.clear()
        iter_bar.close()
        train_tfLogger.close()
        print('----------------------')

if __name__ == '__main__':
    main()