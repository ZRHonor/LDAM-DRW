import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms.transforms import RandomVerticalFlip
import models
from tensorboardX import SummaryWriter
from sklearn.metrics import confusion_matrix
from utils import *
# from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from ImTinyImagenet import ImTinyImagenet
import datetime
from losses import GHMSeesawV2, LDAMLoss, FocalLoss, SeesawLoss, SeesawLoss_prior, GHMcLoss, SoftmaxGHMc, SoftmaxGHMcV2, SoftmaxGHMcV3, SeesawGHMc
from losses import SoftSeesawLoss, GradSeesawLoss_prior, GradSeesawLoss, SoftGradeSeesawLoss, EQLv2, CEloss, EQLloss, GHMSeesawV2

import matplotlib.pyplot as plt

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar100', help='dataset setting')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet32)')
parser.add_argument('--loss_type', default='GHMSeesawV2', type=str, help='loss type')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=200, type=int, help='imbalance factor')
parser.add_argument('--train_rule', default='None', type=str, help='data sampling strategy for train loader')
parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data sampling')
parser.add_argument('--exp_str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--num_classes', dest='num_classes', default=100, type=int)
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--beta', dest='beta', default=1.3, type=float)
best_acc1 = 0


def main():
    args = parser.parse_args()
    args.pretrained=True
    args.num_classes = {'cifar100':100, 'cifar10':10}[args.dataset]
    curr_time = datetime.datetime.now()
    args.store_name = '_'.join([str(curr_time.day), str(curr_time.hour), str(curr_time.minute), 'tinyImagenet', args.arch, args.loss_type, args.train_rule, args.imb_type, str(args.imb_factor), args.exp_str, str(args.seed), str(args.beta)])
    args.imb_factor = 1.0 / args.imb_factor
    print('\n=====================================================================')
    print(args.store_name)
    print('=====================================================================\n')

    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 100 if args.dataset == 'cifar100' else 10
    use_norm = True if args.loss_type in ['LDAM'] else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = ImTinyImagenet(root='data/TinyImageNet/train', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, transform=transform_train)
    val_dataset = datasets.ImageFolder(root='data/TinyImageNet/val', transform=transform_val)
    # if args.dataset == 'cifar10':
    #     train_dataset = IMBALANCECIFAR10(root='./data/CIFAR10', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
    #     val_dataset = datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform_val)
    # elif args.dataset == 'cifar100':
    #     train_dataset = IMBALANCECIFAR100(root='./data/CIFAR100', imb_type=args.imb_type, imb_factor=args.imb_factor, rand_number=args.rand_number, train=True, download=True, transform=transform_train)
    #     val_dataset = datasets.CIFAR100(root='./data/CIFAR100', train=False, download=True, transform=transform_val)
    # else:
    #     warnings.warn('Dataset is not listed')
    #     return
    cls_num_list = train_dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list
    
    train_sampler = None
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    # TAG Init train rule
    if args.train_rule == 'None':
        train_sampler = None  
        per_cls_weights = None 
    elif args.train_rule == 'EffectiveNumber':
        train_sampler = None
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
    elif args.train_rule == 'ClassBlance':
        train_sampler = None  
        per_cls_weights = 1.0 / np.array(cls_num_list)
        per_cls_weights = per_cls_weights/ np.mean(per_cls_weights)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
    elif args.train_rule == 'ClassBlanceV2':
        train_sampler = None  
        per_cls_weights = 1.0 / np.power(np.array(cls_num_list), 0.25)
        per_cls_weights = per_cls_weights/ np.mean(per_cls_weights)
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
    else:
        warnings.warn('Sample rule is not listed')
    
    # TAG Init loss
    if args.loss_type == 'CE':
        # criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        criterion = CEloss(weight=per_cls_weights).cuda(args.gpu)
    elif args.loss_type == 'LDAM':
        criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
    elif args.loss_type == 'Focal':
        criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(args.gpu)
    elif args.loss_type == 'Seesaw':
        criterion = SeesawLoss(num_classes=num_classes).cuda(args.gpu)
    elif args.loss_type == 'GradSeesawLoss':
        criterion = GradSeesawLoss(num_classes=num_classes).cuda(args.gpu)
    elif args.loss_type == 'SoftSeesaw':
        criterion = SoftSeesawLoss(num_classes=num_classes, beta=args.beta).cuda(args.gpu)
    elif args.loss_type == 'SoftGradeSeesawLoss':
        criterion = SoftGradeSeesawLoss(num_classes=num_classes).cuda(args.gpu)
    elif args.loss_type == 'Seesaw_prior':
        criterion = SeesawLoss_prior(cls_num_list=cls_num_list).cuda(args.gpu)
    elif args.loss_type == 'GradSeesawLoss_prior':
        criterion = GradSeesawLoss_prior(cls_num_list=cls_num_list).cuda(args.gpu)
    elif args.loss_type == 'GHMc':
        criterion = GHMcLoss(bins=30, momentum=0.75, use_sigmoid=True).cuda(args.gpu)
    elif args.loss_type == 'SoftmaxGHMc':
        criterion = SoftmaxGHMc(bins=30, momentum=0.75).cuda(args.gpu)
    elif args.loss_type == 'SoftmaxGHMcV2':
        criterion = SoftmaxGHMcV2(bins=30, momentum=0.75).cuda(args.gpu)
    elif args.loss_type == 'SoftmaxGHMcV3':
        criterion = SoftmaxGHMcV3(bins=30, momentum=0.75).cuda(args.gpu)
    elif args.loss_type == 'SeesawGHMc':
        criterion = SeesawGHMc(bins=30, momentum=0.75).cuda(args.gpu)
    elif args.loss_type == 'EQLv2':
        criterion = EQLv2(num_classes=num_classes).cuda(args.gpu)
    elif args.loss_type == 'EQL':
        criterion = EQLloss(cls_num_list=cls_num_list).cuda(args.gpu)
    elif args.loss_type == 'GHMSeesawV2':
        criterion = GHMSeesawV2(num_classes=num_classes, beta=args.beta).cuda(args.gpu)
    else:
        warnings.warn('Loss type is not listed')
        return

    valid_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, log_training, tf_writer)
        # print(criterion.cls_num_list.transpose(1,0))
        
        # evaluate on validation set
        acc1 = validate(val_loader, model, valid_criterion, epoch, args, log_testing, tf_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args, log, tf_writer):
    print(args.store_name)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1, input.size(0))
        top5.update(acc5, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2, 2)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}/{1}][{2}/{3}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, args.epochs, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
            print(output)
            # print('avg_g:{}'.format(criterion.))
            log.write(output + '\n')
            log.flush()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
    # tf_writer.add_histogram('linear.weight', model.linear.weight, epoch)
    # tf_writer.add_histogram('linear.bias', model.linear.bias, epoch)
    # fig = plt.figure()
    # plt.plot(criterion.ratio.cpu().numpy().transpose(1,0))
    # plt.ylim(0.8, 1.2)
    # plt.show()
    # tf_writer.add_figure('ratio_{}'.format(epoch), fig,  epoch)
    
    if args.loss_type in ['GHMc', 'SoftmaxGHMc', 'SoftmaxGHMcV2', 'SoftmaxGHMcV3', 'SeesawGHMc']:
        # bins = len(criterion.acc_sum.tolist())
        limits = np.arange(0,30,1)/30
        accsum = criterion.acc_sum.cpu().numpy()
        accsum = accsum/np.sum(accsum)
        tf_writer.add_histogram_raw(
            'Hist_in_GHM',
            min=0,
            max=1,
            num=0,
            sum=0,
            sum_squares=0,
            bucket_limits=limits.tolist(),  # <- note here.
            bucket_counts=accsum.tolist(),
            global_step=epoch
        )
        accsum = np.log(accsum+1e-8)
        accsum = accsum-np.min(accsum)
        accsum = accsum/np.sum(accsum)
        temp = np.linspace(accsum.max(), accsum.min(), len(accsum))
        accsum = accsum-temp
        tf_writer.add_histogram_raw(
            'LOGHist_in_GHM',
            min=0,
            max=1,
            num=0,
            sum=0,
            sum_squares=0,
            bucket_limits=limits.tolist(),  # <- note here.
            bucket_counts=accsum.tolist(),
            global_step=epoch
        )
    elif args.loss_type in ['Seesaw', 'SoftSeesaw', 'GradSeesawLoss', 'SoftGradeSeesawLoss']:
        limits = np.arange(0,args.num_classes,1)
        accsum = criterion.cls_num_list.cpu().numpy().reshape(-1,)
        accsum = accsum/np.sum(accsum)
        tf_writer.add_histogram_raw(
            'cls_num_list',
            min=0,
            max=args.num_classes,
            num=0,
            sum=0,
            sum_squares=0,
            bucket_limits=limits.tolist(),  # <- note here.
            bucket_counts=accsum.tolist(),
            global_step=epoch
        )
        accsum = np.log(accsum+1e-8)
        accsum = accsum-np.min(accsum)
        accsum = accsum/np.sum(accsum)
        tf_writer.add_histogram_raw(
            'LOGcls_num_list',
            min=0,
            max=args.num_classes,
            num=0,
            sum=0,
            sum_squares=0,
            bucket_limits=limits.tolist(),  # <- note here.
            bucket_counts=accsum.tolist(),
            global_step=epoch
        )

def validate(val_loader, model, criterion, epoch, args, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1, input.size(0))
            top5.update(acc5, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                .format(flag=flag, top1=top1, top5=top5, loss=losses))
        # out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        out_cls_acc = '{} Class Accuracy: \n {}'.format(flag, np.array2string(cls_acc, separator='\t', formatter={'float_kind':lambda x: "%.3f" % x}))
        if args.dataset == 'cifar100':
            temp = {'fre1':np.mean(cls_acc[0:25]),
                    'fre2':np.mean(cls_acc[25:50]),
                    'fre3':np.mean(cls_acc[50:75]),
                    'fre4':np.mean(cls_acc[75:])}
        else:
            temp = {'fre1':np.mean(cls_acc[0:2]),
                    'fre2':np.mean(cls_acc[2:5]),
                    'fre3':np.mean(cls_acc[5:8]),
                    'fre4':np.mean(cls_acc[8:])}
        
        print(output)
        print(out_cls_acc)
        print(temp)
        print('acc_var:{}'.format(np.var(cls_acc)))
        if log is not None:
            log.write(output + '\n')
            log.write(out_cls_acc + '\n')
            log.write(str(temp)+'\n')
            log.write('acc_var:{}\n'.format(np.var(cls_acc)))
            log.flush()

        tf_writer.add_scalar('loss/test_'+ flag, losses.avg, epoch)
        tf_writer.add_scalar('acc_var/test_'+ flag, np.var(cls_acc), epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        
        tf_writer.add_scalars('acc_mean/test_' + flag + '_acc_', temp, epoch)
        tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i):x for i, x in enumerate(cls_acc)}, epoch)
        limits = np.arange(0,args.num_classes,1)
        tf_writer.add_histogram_raw(
            'acc_per_class',
            min=0,
            max=100,
            num=cls_acc.sum()*100,
            sum=0,
            sum_squares=0,
            bucket_limits=limits.tolist(),  # <- note here.
            bucket_counts=(cls_acc*100*0.3).tolist(),
            global_step=epoch
        )
        fig = plt.figure()
        plt.plot(cls_acc)
        plt.ylim(0,1)
        tf_writer.add_figure('total_acc{}'.format(epoch), fig, epoch)

    return top1.avg

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 180:
        lr = args.lr * 0.0001
    elif epoch > 160:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    plt.switch_backend('agg')
    main()