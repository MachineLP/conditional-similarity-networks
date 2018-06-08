from __future__ import print_function
import argparse
import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from visdom import Visdom
import numpy as np
import Resnet_18
from csn import ConditionalSimNet
from PIL import Image


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='./csn_model/checkpoint.pth.tar', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Conditional_Similarity_Network', type=str,
                    help='name of experiment')
parser.add_argument('--embed_loss', type=float, default=5e-3, metavar='M',
                    help='parameter for loss for embedding norm')
parser.add_argument('--mask_loss', type=float, default=5e-4, metavar='M',
                    help='parameter for loss for mask norm')
parser.add_argument('--num_traintriplets', type=int, default=100000, metavar='N',
                    help='how many unique training triplets (default: 100000)')
parser.add_argument('--dim_embed', type=int, default=64, metavar='N',
                    help='how many dimensions in embedding (default: 64)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--learned', dest='learned', action='store_true',
                    help='To learn masks from random initialization')
parser.add_argument('--prein', dest='prein', action='store_true',
                    help='To initialize masks to be disjoint')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                    help='Use visdom to track and plot')
parser.add_argument('--conditions', nargs='*', type=int,
                    help='Set of similarity notions')
parser.set_defaults(test=False)
parser.set_defaults(learned=False)
parser.set_defaults(prein=False)
parser.set_defaults(visdom=False)
args = parser.parse_args()

conditions = [0,1,2,3]

model = Resnet_18.resnet18(pretrained=False, embedding_size=args.dim_embed)
csn_model = ConditionalSimNet(model, n_conditions=len(conditions),
                                  embedding_size=args.dim_embed, learnedmask=args.learned, prein=args.prein)

class lp_net(nn.Module):
    def __init__(self, embeddingnet):
        super(lp_net, self).__init__()
        self.embeddingnet = embeddingnet
    def forward(self, x, c):
        embedded, masknorm_norm, embed_norm, tot_embed_norm = self.embeddingnet(x, c)
        return embedded, masknorm_norm, embed_norm, tot_embed_norm
t_net = lp_net(csn_model)


# optionally resume from a checkpoint
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        loaded_state_dict = checkpoint['state_dict']
        state = t_net.state_dict()
        # print ('>>>>>>>>>', state)
        for k in loaded_state_dict:
            # print (k)
            if k in state:
                print (k)
                state[k] = loaded_state_dict[k]
        t_net.load_state_dict(state)
        #print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

def default_loader(path):
    # 注意要保证每个batch的tensor大小时候一样的。
    return Image.open(path).convert('RGB')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform=transforms.Compose([
                              transforms.Resize(112),
                              transforms.CenterCrop(112),
                              transforms.ToTensor(),
                              normalize,
                              ])

def torch_inference(img_path, c=0):

    img = default_loader(img_path)
    # 要转为tensor
    c = torch.from_numpy(np.asarray([c]))
    # 三通道转为四通道
    img = transform(img).unsqueeze_(0)
    print (img)
    with torch.no_grad():
        embedded, masknorm_norm, embed_norm, tot_embed_norm = t_net(img, c)
        print (embedded)
    return embedded[0]

if __name__ == '__main__':
    img_path = 'lp.jpg'
    torch_inference = torch_inference(img_path)

'''
class CS_Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(CS_Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z, c):
        """ x: Anchor image,
            y: Distant (negative) image,
            z: Close (positive) image,
            c: Integer indicating according to which notion of similarity images are compared"""
        embedded_x, masknorm_norm_x, embed_norm_x, tot_embed_norm_x = self.embeddingnet(x, c)
        embedded_y, masknorm_norm_y, embed_norm_y, tot_embed_norm_y = self.embeddingnet(y, c)
        embedded_z, masknorm_norm_z, embed_norm_z, tot_embed_norm_z = self.embeddingnet(z, c)
        mask_norm = (masknorm_norm_x + masknorm_norm_y + masknorm_norm_z) / 3
        embed_norm = (embed_norm_x + embed_norm_y + embed_norm_z) / 3
        mask_embed_norm = (tot_embed_norm_x + tot_embed_norm_y + tot_embed_norm_z) / 3
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, mask_norm, embed_norm, mask_embed_norm'''


