import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _expand_onehot_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(
        (labels >= 0) & (labels < label_channels), as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1
    bin_label_weights = label_weights.view(-1, 1).expand(
        label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)

class GHMcLoss(nn.Module):
    def __init__(self, bins=10, momentum=0, use_sigmoid=True, loss_weight=1.0):
        super(GHMcLoss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.use_sigmoid = use_sigmoid
        # if not self.use_sigmoid:
        #     raise NotImplementedError
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        label_weight = torch.ones_like(target)
        if pred.dim() != target.dim():
            target, label_weight = _expand_onehot_labels(
                target, label_weight, pred.size(-1))
        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        g = torch.abs(pred.sigmoid().detach() - target)

        valid = label_weight > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n
        if self.use_sigmoid:
            loss = F.binary_cross_entropy_with_logits(
                pred, target, weights, reduction='sum') / tot
        else:
            # TODO GHMc without 
            loss = F.cross_entropy(input, target, reduction='none')
            
            # loss = F.binary_cross_entropy_with_logits(
            #     pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight

class GroupGHMcLoss(nn.Module):
    # TODO GroupGHMcLoss
    def __init__(self, weight=None, gamma=0.):
        super(GroupGHMcLoss, self).__init__()
        pass

    def forward(self, x, target):
        return 0

class SeesawLoss_prior(nn.Module):
    def __init__(self, cls_num_list, p=0.8):
        super(SeesawLoss_prior, self).__init__()
        self.num_classes = len(cls_num_list)
        self.cls_num_list = np.array(cls_num_list).reshape(self.num_classes, 1)

        self.weight_matrix = (1.0 / self.cls_num_list) * self.cls_num_list.transpose()
        self.weight_matrix[self.weight_matrix>1] = 1  
        self.weight_matrix = np.power(self.weight_matrix, p)

        self.weight_matrix = torch.FloatTensor(self.weight_matrix).cuda()


    def forward(self, x, target):
        '''
        x: b * C
        '''
        bs = x.shape[0]
        target_onehot = F.one_hot(target, num_classes=self.num_classes).float()
        weight = torch.mm(target_onehot, self.weight_matrix)
        weighted_x = x + torch.log(weight)
        softmax_x = F.softmax(weighted_x, 1)
        
        loss = -torch.sum(target_onehot * torch.log(softmax_x))/bs
        return loss

class SeesawLoss(nn.Module):
    def __init__(self, num_classes,p=0.8):
        super(SeesawLoss, self).__init__()
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
        num_classes_batch = torch.sum(target_onehot, 0, keepdim=True).detach().cpu().permute(1,0)
        self.cls_num_list += num_classes_batch
        weight_matrix = self.get_weight_matrix()
        weight = torch.mm(target_onehot, weight_matrix)
        weighted_x = x + torch.log(weight)
        softmax_x = F.softmax(weighted_x, 1)
        
        loss = -torch.sum(target_onehot * torch.log(softmax_x))/bs
        return loss
