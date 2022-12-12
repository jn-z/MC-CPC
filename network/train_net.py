import torch
import torch.nn as nn
import torch.nn.init as init
import math
import torch.nn.functional as F
import numpy as np
import math
import copy
from torch.autograd import Variable
from module.utils import BatchEndParam
from network.network_componet import fullyConnect, ResNet
import pdb
class TrainModule(nn.Module):
    def __init__(self, args):

        super(TrainModule, self).__init__()
        self.args = args

        self.layer = nn.Sequential()
        self.layer.add_module('resnet', ResNet(layers=[3, 4, 6, 3], flatten_dim=args.flatten_dim, spkVec_dim=self.args.cvector_dim))
        self.marginType = args.marginType
        self.layer.add_module('header', nn.ModuleList(
                [fullyConnect(target_num=self.args.spk_num, spkVec_dim=self.args.cvector_dim)]))

    def forward(self, src, state_lab, total_fea_frames, npart, is_train=True):
        anchor_data = src
        anchor_sv = self.layer.resnet(anchor_data)
        tar = self.layer.header[npart](anchor_sv)
        if is_train:
            tar_select_new = torch.gather(tar, 1, state_lab)
            ce_loss = -torch.log(tar_select_new + pow(10.0, -8))
            return anchor_sv, tar_select_new, ce_loss, tar
        else:
            return tar