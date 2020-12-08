import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GradSeesawLoss(nn.Module):
    def __init__(self, num_classes, p=0.8):
        super(GradSeesawLoss, self).__init__()
        self.num_classes = num_classes
        cls_num_list = torch.ones(size=(num_classes,1)) * 1e-6
        self.register_buffer('cls_num_list', cls_num_list)
        self.p = p

        
    @torch.no_grad()
    def get_weight_matrix(self):
        weight_matrix = (1.0 / self.cls_num_list) * self.cls_num_list.transpose(1,0)
        weight_matrix[weight_matrix>1] = 1  
        weight_matrix = torch.pow(weight_matrix, self.p)

        weight_matrix = weight_matrix.cuda()
        return weight_matrix

    def forward(self, x, target):
        '''
        x: b * C
        '''
        bs = x.shape[0]
        target_onehot = F.one_hot(target, num_classes=self.num_classes).float()
        g = torch.abs(x.sigmoid().detach() - target_onehot)
        # num_classes_batch = torch.sum(target_onehot*(1-g), 0, keepdim=True).detach().cpu().permute(1,0)
        num_classes_batch = torch.sum(target_onehot, 0, keepdim=True).detach().cpu().permute(1,0)
        self.cls_num_list += num_classes_batch
        weight_matrix = self.get_weight_matrix()
        weight = torch.mm(target_onehot, weight_matrix)
        weighted_x = x + torch.log(weight + g)
        softmax_x = F.softmax(weighted_x, 1)
        
        loss = -torch.sum(target_onehot * torch.log(softmax_x)) / bs
        return loss
