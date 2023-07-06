import skimage
from skimage.segmentation import slic, felzenszwalb, quickshift
from torchvision import transforms as transforms
from tqdm import tqdm

import CR
import curves
import network
import utils
import random
import argparse
import numpy as np
from metrics import StreamSegMetrics
from torch.utils import data
from datasets import Cityscapes
from utils import ext_transforms as et
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import sys
import time
import lowlight_model
import Myloss
from torchvision import transforms
import torch
import torch.nn as nn
from utils.visualizer import Visualizer
from PIL import Image, ImageStat
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os,sys
import random
import shutil
import math
from tensorboardX import SummaryWriter

transf = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])



def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data/cityscapes',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Train Options

    parser.add_argument("--total_itrs", type=int, default=14e3,
                        help="epoch number")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=384)

    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='28333',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument('--lowlight_lr', type=float, default=0.0001)
    parser.add_argument('--lowlight_weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--snapshots_folder', type=str, default="./results_new/")
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    分别针对训练集、验证集、测试集做三种数据增强变换
    """
    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            #et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
        ])
        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
    return train_dst


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def GetNeg(negnumber):
    #generate positive samples
    transff = transforms.ToTensor()
    pathGT = "./datasets/data/Contrast/GT/"
    pathDirGT = os.listdir(pathGT)
    sampleL = random.sample(pathDirGT, negnumber)
    adjust_images = []
    for i in range(0, negnumber):
        X_L = Image.open(pathGT + sampleL[i])
        X_L = X_L.resize((384, 384), Image.ANTIALIAS)
        ctrlnum = random.randint(0, 10)

        X_L = np.array(X_L)
        if (ctrlnum % 3 == 0):
            X_L = curves.ImageBrightnessAdjuster.darken_image_tone_mapping(X_L)
        elif(ctrlnum % 3 == 1):
            X_L = curves.ImageBrightnessAdjuster.gamma_correction(X_L)
        else:
            X_L = curves.ImageBrightnessAdjuster.log_curve(X_L)

        X_L = transff(X_L)

        adjust_images.append(X_L)
    L = torch.stack(adjust_images, dim=0)
    return  L.cuda()
def GetPos(PosNum):
    GT = []
    for i in range(PosNum):
        transff = transforms.ToTensor()
        pathGT = "./datasets/data/Contrast/GT/"
        pathDirGT = os.listdir(pathGT)
        X_GT = Image.open(pathGT + random.sample(pathDirGT, 1)[0])
        X_GT = X_GT.resize((384, 384), Image.ANTIALIAS)
        X_GT = transff(X_GT)
        GT.append(X_GT)

    return torch.stack(GT, dim=0).cuda()
def Segmentation(enhanced_image):
    imgseg = enhanced_image.cpu().detach().numpy().transpose(0, 2, 3, 1)
    segments = []
    for i in range(imgseg.shape[0]):
        segment = felzenszwalb(imgseg[i], scale=100, sigma=0.5, min_size=300)
        segment = torch.tensor(segment).cuda()
        segments.append(segment)
    segment = torch.stack(segments)
    return segment

def main():
    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    # Setup dataloader
    if opts.dataset=='voc' and not opts.crop_val:
        opts.val_batch_size = 1
    train_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=4)

    cur_itrs = 0
    cur_epochs = 0
    #==========   Train Loop   ==========#


    #lowlight
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    DCE_net = lowlight_model.enhance_net_nopool().cuda()
    DCE_net.apply(weights_init)
    L_color = Myloss.L_color()
    L_con = Myloss.L_con()
    L_TV = Myloss.L_TV()
    L_segexp = Myloss.L_segexp()
    L_percept = Myloss.perception_loss()
    L_con_neg = Myloss.L_con_neg()
    L_infonce = CR.InfoNceLoss()


    lowlight_optimizer = torch.optim.Adam(DCE_net.parameters(), lr=opts.lowlight_lr, weight_decay=opts.lowlight_weight_decay)

    picknumber = 2
    negnumber = 2
    cnt = 1

    iteration = 0
    epoch = 0
    while True: #cur_itrs < opts.total_itrs:
        # =====  Train  =====
        DCE_net.train()
        cur_epochs += 1

        for (images, labels) in train_loader:
            iteration += 1
            if len(images) <=1:
                continue
            cur_itrs += 1


            L = GetNeg(negnumber)
            GT = GetPos(picknumber)
            images = images.to(device, dtype=torch.float32)
            enhanced_image_1,enhanced_image,A  = DCE_net(images)
            segment = Segmentation(enhanced_image)

            Loss_TV = 200*L_TV(A)
            loss_col = 8*torch.mean(L_color(enhanced_image))
            loss_segexp = torch.mean(L_segexp(enhanced_image, segment))
            loss_percent = torch.mean(L_percept(images,enhanced_image))
            loss_cont = 10 * torch.mean(max(L_con(enhanced_image, GT) - L_con_neg(L, enhanced_image) + 0.3,L_con(enhanced_image, GT) - L_con(enhanced_image, GT)))
            loss_cont2 = 10*torch.mean(max(L_infonce(enhanced_image, GT , L),L_con(enhanced_image, GT) - L_con(enhanced_image, GT)))
            lowlight_loss = Loss_TV + loss_col  + loss_percent + loss_segexp + loss_cont2 + loss_cont
            print(lowlight_loss)


            lowlight_optimizer.zero_grad()
            lowlight_loss.backward()
            torch.nn.utils.clip_grad_norm(DCE_net.parameters(),opts.grad_clip_norm)
            lowlight_optimizer.step()
            print("---------------epoch:" + str(epoch) + "  cnt:" + str(cnt) + "---------------")
            cnt=cnt+1

            
            if (cnt % 1300)==0:
                epoch = epoch + 1 
                torch.save(DCE_net.state_dict(), opts.snapshots_folder + "Epoch" + str(cnt/1300) + '.pth')

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()
