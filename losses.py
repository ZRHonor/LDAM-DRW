# import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

def weighted_softmax(x, target, weight):
    exp_x = torch.exp(x)
    all_sum = torch.sum(exp_x * weight, dim=1)
    pred = torch.sum(exp_x * target, dim=1) / all_sum
    loss = -torch.mean(torch.log(pred))
    return loss


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
        ori_target = target
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
            loss = F.cross_entropy(pred, ori_target, reduction='none')

            # loss = F.binary_cross_entropy_with_logits(
            #     pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight

class SoftmaxGHMc(nn.Module):
    def __init__(self, bins=10, momentum=0, use_sigmoid=False, loss_weight=1.0):
        super(SoftmaxGHMc, self).__init__()
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
        weights = torch.zeros_like(target).float()
        values = F.cross_entropy(pred, target, reduction='none')
        values_clip = values.clone().detach()
        values_clip = values_clip / torch.max(values_clip)
        # values_clip[values_clip>1]=1-1e-6
        # valid = (values>0)
        tot = values.shape[0]
        n = 0

        edges = self.edges
        mmt = self.momentum

        for i in range(self.bins):
            inds = (values_clip >= edges[i]) & (values_clip < edges[i + 1])
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
        weights /= tot
        loss = weights * values
        return loss.sum()

class SoftmaxGHMcV2(nn.Module):
    def __init__(self, bins=10, momentum=0, use_sigmoid=False, loss_weight=1.0):
        super(SoftmaxGHMcV2, self).__init__()
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
        weights = torch.zeros_like(target).float()
        values = F.cross_entropy(pred, target, reduction='none')
        values_clip = values.clone().detach()
        values_clip = 1 - torch.exp(-values_clip)
        # values_clip = values_clip / torch.max(values_clip)
        # values_clip[values_clip>1]=1-1e-6
        # valid = (values>0)
        tot = values.shape[0]
        n = 0

        edges = self.edges
        mmt = self.momentum

        for i in range(self.bins):
            inds = (values_clip >= edges[i]) & (values_clip < edges[i + 1])
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
        weights /= tot
        loss = weights * values
        return loss.sum()

class SoftmaxGHMcV3(nn.Module):
    def __init__(self, bins=10, momentum=0, use_sigmoid=False, loss_weight=1.0):
        super(SoftmaxGHMcV3, self).__init__()
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
        weights = torch.zeros_like(target).float()

        target_onehot = F.one_hot(target.clone().detach(), pred.shape[1])

        g = torch.abs(F.softmax(pred, dim=1) - target_onehot)*target_onehot
        g = g.sum(1)

        tot = g.shape[0]
        n = 0

        edges = self.edges
        mmt = self.momentum

        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1])
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
        weights /= tot
        loss = weights * F.cross_entropy(pred, target, reduction='none')
        return loss.sum()

class GroupGHMcLoss(nn.Module):
    # TODO GroupGHMcLoss
    def __init__(self, weight=None, gamma=0.):
        super(GroupGHMcLoss, self).__init__()
        pass

    def forward(self, x, target):
        return 0

class SeesawLoss_prior(nn.Module):
    def __init__(self, cls_num_list, p=0.8, num_classes=100):
        super(SeesawLoss_prior, self).__init__()
        self.num_classes = len(cls_num_list)
        cls_num_list = np.array(cls_num_list).reshape(self.num_classes, 1)

        weight_matrix = (1.0 / cls_num_list) * cls_num_list.transpose()
        weight_matrix[weight_matrix>1] = 1  
        weight_matrix = np.power(weight_matrix, p)

        weight_matrix = torch.FloatTensor(weight_matrix)
        
        self.register_buffer('weight_matrix', weight_matrix)


        self.num_classes = num_classes
        neg_grad = torch.zeros(size=(1, num_classes)) + 1e-6
        pos_grad = torch.zeros(size=(1, num_classes)) + 1e-6
        self.register_buffer('neg_grad', neg_grad)
        self.register_buffer('pos_grad', pos_grad)
        self.ratio = torch.zeros(size=(1, num_classes))

    def forward(self, x, target):
        '''
        x: b * C
        '''
        bs = x.shape[0]
        target_onehot = F.one_hot(target, num_classes=self.num_classes).float()
        weight = torch.mm(target_onehot, self.weight_matrix)
        weighted_x = x + torch.log(weight)
        softmax_x = F.softmax(weighted_x, 1)

        target_onehot = F.one_hot(target, num_classes=self.num_classes).float().detach()
        grad = torch.abs(softmax_x.clone().detach() - target_onehot.clone().detach())
        self.neg_grad += (grad * (1 - target_onehot)).sum(0)
        self.pos_grad += (grad * target_onehot).sum(0)
        self.ratio = self.pos_grad / self.neg_grad
        # print(ratio)
        
        loss = -torch.sum(target_onehot * torch.log(softmax_x))/bs
        return loss

class SeesawLoss(nn.Module):
    def __init__(self, num_classes, p=0.8):
        super(SeesawLoss, self).__init__()
        self.num_classes = num_classes
        cls_num_list = torch.ones(size=(num_classes,1)) * 1e-6
        self.register_buffer('cls_num_list', cls_num_list)
        # self.cls_num_list.cuda()
        self.p = p

        self.num_classes = num_classes
        neg_grad = torch.zeros(size=(1, num_classes)) + 1e-6
        pos_grad = torch.zeros(size=(1, num_classes)) + 1e-6
        self.register_buffer('neg_grad', neg_grad)
        self.register_buffer('pos_grad', pos_grad)
        self.ratio = torch.zeros(size=(1, num_classes))

        
    @torch.no_grad()
    def get_weight_matrix(self):
        weight_matrix = (1.0 / self.cls_num_list) * self.cls_num_list.transpose(1,0)
        weight_matrix[weight_matrix>1] = 1  
        weight_matrix = torch.pow(weight_matrix, self.p)
        # weight_matrix[-1,:] = torch.mean(weight_matrix[:,-1], dim=0)
        # weight_matrix[-1,:] = 1
        # weight_matrix[:,-1] = 1
        # weight_matrix[1,:] = torch.mean(weight_matrix[1:,:])
        # weight_matrix[-1,:] = torch.mean(weight_matrix[:-1,:])
        return weight_matrix

    def forward(self, x, target):
        '''
        x: b * C
        '''
        bs = x.shape[0]
        target_onehot = F.one_hot(target, num_classes=self.num_classes).float()
        # g = torch.abs(x.sigmoid().detach() - target_onehot)
        # num_classes_batch = torch.sum(target_onehot*(1-g), 0, keepdim=True).detach().cpu().permute(1,0)
        num_classes_batch = torch.sum(target_onehot, 0, keepdim=True).detach().permute(1,0)
        self.cls_num_list += num_classes_batch
        weight_matrix = self.get_weight_matrix()
        weight = torch.mm(target_onehot, weight_matrix)
        weighted_x = x + torch.log(weight)
        softmax_x = F.softmax(weighted_x, 1)

        target_onehot = F.one_hot(target, num_classes=self.num_classes).float().detach()
        grad = torch.abs(softmax_x.clone().detach() - target_onehot.clone().detach())
        self.neg_grad += (grad * (1 - target_onehot)).sum(0)
        self.pos_grad += (grad * target_onehot).sum(0)
        self.ratio = self.pos_grad / self.neg_grad
        
        loss = -torch.sum(target_onehot * torch.log(softmax_x))/bs
        return loss

class SeesawLossv2(nn.Module):
    def __init__(self, num_classes, p=0.8):
        super(SeesawLossv2, self).__init__()
        self.num_classes = num_classes
        cls_num_list = torch.ones(size=(num_classes,1)) * 1e-6
        self.register_buffer('cls_num_list', cls_num_list)
        # self.cls_num_list.cuda()
        self.p = p

    @torch.no_grad()
    def get_weight(self, pred, target):
        M_matrix = (1.0 / self.cls_num_list) * self.cls_num_list.transpose(1,0)
        M_matrix[M_matrix>1] = 1  
        M_matrix = torch.pow(M_matrix, self.p)
        M_matrix = torch.mm(target, M_matrix)
        
        pred_correct = torch.sum(pred*target, dim=1, keepdim=True)
        C_matrix = pred / pred_correct
        C_matrix[C_matrix>1] = 1
        C_matrix = torch.pow(C_matrix, self.q)
        return M_matrix * C_matrix
    
    def forward(self, x, target):
        # bs = x.shape[0]
        target_onehot = F.one_hot(target, num_classes=self.num_classes).float().detach()
        pred = F.softmax(x, 1).detach()
        num_classes_batch = torch.sum(target_onehot, 0, keepdim=True).detach().permute(1,0)
        self.cls_num_list += num_classes_batch
        weight = self.get_weight(pred, target_onehot)
        loss = weighted_softmax(x, target_onehot, weight)
        return loss

    

class SoftSeesawLoss(nn.Module):
    def __init__(self, num_classes, p=0.8, beta=0.5):
        super(SoftSeesawLoss, self).__init__()
        self.num_classes = num_classes
        cls_num_list = torch.ones(size=(num_classes,1)) * 1e-6
        self.register_buffer('cls_num_list', cls_num_list)
        # self.cls_num_list.cuda()
        self.p = p
        self.beta = beta
        self.avg_g = 0
        self.alpha = 0.7

        self.num_classes = num_classes
        neg_grad = torch.zeros(size=(1, num_classes)) + 1e-6
        pos_grad = torch.zeros(size=(1, num_classes)) + 1e-6
        self.register_buffer('neg_grad', neg_grad)
        self.register_buffer('pos_grad', pos_grad)
        self.ratio = torch.zeros(size=(1, num_classes))

        
    @torch.no_grad()
    def get_weight_matrix(self):
        weight_matrix = (1.0 / self.cls_num_list) * self.cls_num_list.transpose(1,0)
        weight_matrix[weight_matrix>1] = 1  
        weight_matrix = torch.pow(weight_matrix, self.p)
        # weight_matrix[-1,:] = torch.mean(weight_matrix[:,-1], dim=0)
        # weight_matrix[-1,:] = weight_matrix.mean(dim=1)
        # weight_matrix[:,-1] = 1
        # weight_matrix[-1,:] = torch.mean(weight_matrix[:-1,:])
        # weight_matrix[1,:] = torch.mean(weight_matrix[1:,:])
        # weight_matrix[weight_matrix<1e-6] = 1e-6
        # weight_matrix = torch.clip(weight_matrix, 1e-6, 1)
        return weight_matrix

    def forward(self, x, target):
        '''
        x: b * C
        '''
        bs = x.shape[0]
        target_onehot = F.one_hot(target, num_classes=self.num_classes).float()
        g = torch.abs(x.sigmoid().detach() - target_onehot)
        g_of_samples = torch.sum(target_onehot*g, 1)
        self.avg_g = self.alpha*self.avg_g + (1-self.alpha)*g_of_samples.mean()
        num_classes_batch = torch.sum(target_onehot*(1+(self.avg_g - g)), 0, keepdim=True).detach().permute(1,0)
        # num_classes_batch = torch.sum(target_onehot*(1+(0.5 - g)), 0, keepdim=True).detach().permute(1,0)
        # self.avg_g = self.alpha*self.avg_g + (1-self.alpha)*num_classes_batch.mean()
        # num_classes_batch += self.avg_g
        # num_classes_batch = torch.sum(target_onehot*g, 0, keepdim=True).detach().permute(1,0)
        self.cls_num_list += num_classes_batch
        weight_matrix = self.get_weight_matrix()
        weight = torch.mm(target_onehot, weight_matrix)
        weighted_x = x + torch.log(weight)
        # weighted_x = x + torch.log(weight)
        softmax_x = F.softmax(weighted_x, 1)
        
        target_onehot = F.one_hot(target, num_classes=self.num_classes).float().detach()
        grad = torch.abs(softmax_x.clone().detach() - target_onehot.clone().detach())
        self.neg_grad += (grad * (1 - target_onehot)).sum(0)
        self.pos_grad += (grad * target_onehot).sum(0)
        self.ratio = self.pos_grad / self.neg_grad

        loss = -torch.sum(target_onehot * torch.log(softmax_x))/bs
        return loss

class SeesawGHMc(nn.Module):
    def __init__(self, bins=10, momentum=0, loss_weight=1.0):
        super(SeesawGHMc, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.loss_weight = loss_weight

    def forward(self, x, target):
        weights = torch.zeros_like(x)
        target_onehot = F.one_hot(target, num_classes=x.shape[1]).float()
        edges = self.edges
        mmt = self.momentum

        # gradient length
        g = torch.abs(x.sigmoid().detach() - target_onehot)
        tot = 1
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1])
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
        weights /= tot
        weights = 1.0/torch.div(weights,(weights*target_onehot).sum(1).reshape(-1,1))
        weights[weights>1]=1
        weighted_x = x + torch.log(weights)
        softmax_x = F.softmax(weighted_x, 1)
        
        loss = -torch.sum(target_onehot * torch.log(softmax_x))/x.shape[0]
        return loss

class GradSeesawLoss_prior(nn.Module):
    def __init__(self, cls_num_list, p=0.8, bins=10, momentum=0):
        super(GradSeesawLoss_prior, self).__init__()
        self.num_classes = len(cls_num_list)
        cls_num_list = np.array(cls_num_list).reshape(self.num_classes, 1)

        weight_matrix = (1.0 / cls_num_list) * cls_num_list.transpose()
        weight_matrix[weight_matrix>1] = 1  
        weight_matrix = np.power(weight_matrix, p)

        weight_matrix = torch.FloatTensor(weight_matrix).cuda()
        self.register_buffer('weight_matrix', weight_matrix)


    def forward(self, x, target):
        bs = x.shape[0]
        target_onehot = F.one_hot(target, num_classes=self.num_classes).float()
        g = torch.abs(x.sigmoid().detach() - target_onehot)
        weight = torch.mm(target_onehot, self.weight_matrix)
        weighted_x = x + torch.log(weight * (1+g) )
        softmax_x = F.softmax(weighted_x, 1)
        
        loss = -torch.sum(target_onehot * torch.log(softmax_x))/bs
        return loss

class BlancedSoftmax(nn.Module):
    def __init__(self, cls_num_list, p=0.25):
        super(BlancedSoftmax, self).__init__()
        self.num_classes = len(cls_num_list)
        self.cls_num_list = np.array(cls_num_list).reshape(1, self.num_classes)
        self.cls_num_list = torch.from_numpy(self.cls_num_list).detach()
        self.weight = p * torch.log(self.cls_num_list)

    def forward(self, x, target):
        '''
        x: b * C
        '''
        bs = x.shape[0]
        target_onehot = F.one_hot(target, num_classes=self.num_classes).float()
        weighted_x = x + self.weight
        softmax_x = F.softmax(weighted_x, 1)
        
        loss = -torch.sum(target_onehot * torch.log(softmax_x)) / bs
        return loss

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
        num_classes_batch = torch.sum(target_onehot, 0, keepdim=True).detach().permute(1,0)
        self.cls_num_list += num_classes_batch
        weight_matrix = self.get_weight_matrix()
        weight = torch.mm(target_onehot, weight_matrix)
        weighted_x = x + torch.log(weight * (1+g) )
        softmax_x = F.softmax(weighted_x, 1)
        
        loss = -torch.sum(target_onehot * torch.log(softmax_x)) / bs
        return loss

class SoftGradeSeesawLoss_backup(nn.Module):
    def __init__(self, num_classes, p=0.8):
        super(SoftGradeSeesawLoss_backup, self).__init__()
        self.num_classes = num_classes
        cls_num_list = torch.ones(size=(num_classes,1)) * 1e-6
        self.register_buffer('cls_num_list', cls_num_list)
        self.p = p
    
    @torch.no_grad()
    def get_weight_matrix(self):
        weight_matrix = (1.0 / self.cls_num_list) * self.cls_num_list.transpose(1,0)
        weight_matrix[weight_matrix>1] = 1  
        weight_matrix = torch.pow(weight_matrix, self.p)
        return weight_matrix

    def forward(self, x, target):
        bs = x.shape[0]
        target_onehot = F.one_hot(target, num_classes=self.num_classes).float()
        g = torch.abs(x.sigmoid().detach() - target_onehot)
        num_classes_batch = torch.sum(target_onehot*(1-g), 0, keepdim=True).detach().permute(1,0)
        self.cls_num_list += num_classes_batch
        weight_matrix = self.get_weight_matrix()
        weight = torch.mm(target_onehot, weight_matrix)
        weighted_x = x + torch.log(weight * (1+g))
        softmax_x = F.softmax(weighted_x, 1)
        loss = -torch.sum(target_onehot * torch.log(softmax_x)) / bs
        return loss

class SoftGradeSeesawLoss(nn.Module):
    def __init__(self, num_classes, p=0.8, q=2):
        super(SoftGradeSeesawLoss, self).__init__()
        self.num_classes = num_classes
        # cls_num_list = torch.ones(size=(num_classes,1)) * 1e-6
        cls_num_list = torch.zeros(size=(num_classes,1)) + 1e-6
        self.register_buffer('cls_num_list', cls_num_list)
        self.p = p
        self.q = q
    
    @torch.no_grad()
    def get_weight(self, pred, target):
        M_matrix = (1.0 / self.cls_num_list) * self.cls_num_list.transpose(1,0)
        M_matrix[M_matrix>1] = 1  
        M_matrix = torch.pow(M_matrix, self.p)
        M_matrix = torch.mm(target, M_matrix)
        
        pred_correct = torch.sum(pred*target, dim=1, keepdim=True)
        C_matrix = pred / pred_correct
        C_matrix[C_matrix>1] = 1
        C_matrix = torch.pow(C_matrix, self.q)
        return M_matrix * C_matrix

    def forward(self, x, target):
        bs = x.shape[0]
        target_onehot = F.one_hot(target, num_classes=self.num_classes).float().detach()
        pred = F.softmax(x, 1).detach()
        g = torch.abs(pred - target_onehot)
        # num_classes_batch = torch.sum(target_onehot*(1-g), 0, keepdim=True).detach().permute(1,0)
        # num_classes_batch = torch.sum(target_onehot*g, 0, keepdim=True).detach().permute(1,0)
        num_classes_batch = torch.sum(target_onehot, 0, keepdim=True).detach().permute(1,0)
        self.cls_num_list += num_classes_batch
        weight = self.get_weight(pred, target_onehot)
        # weighted_x = x + torch.log(weight)
        # softmax_x = F.softmax(weighted_x, 1)
        # loss = -torch.sum(target_onehot * torch.log(softmax_x)) / bs
        loss = weighted_softmax(x, target_onehot, weight)
        if torch.isinf(loss):
            loss = weighted_softmax(x, target_onehot, weight)
            sys.exit(0)
        # print('x is nan:{}\nweight is nan:{}'.format(torch.isnan(x).any(), torch.isnan(weight).any()))
        if (torch.isnan(x).any() or torch.isnan(weight).any()):
            sys.exit(0)
        return loss

class EQLv2(nn.Module):
    def __init__(self, num_classes, gamma=12, alpha=4, mu=0.8):
        super(EQLv2, self).__init__()
        self.num_classes = num_classes
        neg_grad = torch.zeros(size=(1, num_classes)) + 1e-6
        pos_grad = torch.zeros(size=(1, num_classes)) + 1e-6
        self.register_buffer('neg_grad', neg_grad)
        self.register_buffer('pos_grad', pos_grad)
        self.gamma = gamma
        self.alpha = alpha
        self.mu = mu

    @torch.no_grad()
    def get_weight(self, target):
        ratio_g = self.pos_grad / self.neg_grad
        r = 1 / (1 + torch.exp(-self.gamma*(ratio_g - self.mu)))
        q = 1 + self.alpha * (1 - r)
        return q*target + r*(1-target)

    def forward(self, x, target):
        bs = x.shape[0]
        target_onehot = F.one_hot(target, num_classes=self.num_classes).float().detach()
        pred = x.sigmoid()

        weight = self.get_weight(target_onehot)

        grad = weight * torch.abs(F.softmax(x, 1).detach() - target_onehot)
        self.neg_grad += (grad * (1 - target_onehot)).sum(0)
        self.pos_grad += (grad * target_onehot).sum(0)

        # log_pred = torch.log(torch.abs())
        loss = F.binary_cross_entropy(pred, target_onehot, weight=weight, reduction='sum')
        return loss/bs
        # loss = -torch.log(grad*weight).()
        # loss = torch.FloatTensor([0])
        # log_pred = torch.log(1-grad)*target_onehot
        # return -log_pred
        # loss = (-weight*log_pred).sum()/bs
        # loss = 
        # return (weight*grad).mean()

class CEloss(nn.Module):
    def __init__(self, weight=None, num_classes=100):
        super(CEloss, self).__init__()
        self.weight=weight

        self.num_classes = num_classes
        neg_grad = torch.zeros(size=(1, num_classes)) + 1e-6
        pos_grad = torch.zeros(size=(1, num_classes)) + 1e-6
        self.register_buffer('neg_grad', neg_grad)
        self.register_buffer('pos_grad', pos_grad)
        self.ratio = torch.zeros(size=(1, num_classes))
        self.mm = 0.9

    def forward(self, x, target):
        target_onehot = F.one_hot(target, num_classes=self.num_classes).float().detach()
        grad = torch.abs(F.softmax(x, 1).detach() - target_onehot)
        # self.neg_grad = self.neg_grad*self.mm + (1-self.mm) * (grad * (1 - target_onehot)).sum(0)
        # self.pos_grad = self.pos_grad*self.mm + (1-self.mm) * (grad * target_onehot).sum(0)
        self.neg_grad += (grad * (1 - target_onehot)).sum(0)
        self.pos_grad += (grad * target_onehot).sum(0)
        self.ratio = self.pos_grad / self.neg_grad
        # print(ratio)
        return F.cross_entropy(x, target, self.weight)


class EQLloss(nn.Module):
    def __init__(self, cls_num_list, r=3e-3):
        super(EQLloss, self).__init__()
        self.cls_num_list = cls_num_list
        self.r = r
        self.num_classes = len(cls_num_list)

        total = np.asarray(cls_num_list).sum()
        weight_matrix = torch.ones(size=(self.num_classes, self.num_classes))
        sum=0
        div = self.num_classes-1
        for i in range(self.num_classes):
            sum+=cls_num_list[div]
            if sum >= (total*self.r):
                break
            div-=1
        
        weight_matrix[div:,div:] = torch.eye(self.num_classes-div)
        self.register_buffer('weight_matrix', weight_matrix)

    def forward(self, x, target):
        bs = x.shape[0]
        x = x.sigmoid()
        target_onehot = F.one_hot(target, num_classes=self.num_classes).float().detach()
        weight = torch.mm(target_onehot, self.weight_matrix)
        # cls_loss =  F.binary_cross_entropy(x.view(-1), target_onehot.view(-1), weight=weight.view(-1), reduction='none')
        cls_loss = F.binary_cross_entropy(x.view(-1), target_onehot.view(-1), reduction='none')
        return torch.sum(cls_loss) / bs
