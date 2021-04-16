# test DBcloud
# push from mix

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
import os
import shutil
from torchvision import datasets
from tqdm import tqdm
# import json
from PIL import Image
import random

# extra_imgs = os.listdir('data/background/')
# count = 467
# for img in extra_imgs:
#     try:
#         image = Image.open('data/background/'+img).convert('RGB')
#         count += 1
#         image.save('data/New101/train/BACKGROUND_Google/image_{:0>4d}.jpg'.format(count))
#     except:
#         print(img)

# print('done')


# class_numbers = []
# allcates = os.listdir('data/New101/train')
# for cate in allcates:
#     class_numbers.append([cate, len(os.listdir('data/New101/train/'+cate))])

# class_numbers.sort(key=lambda x:x[1], reverse=True)
# for cate in class_numbers:
#     os.rename('data/New101/train/'+cate[0], 'data/New101/train/{:0>4d}_{}'.format(cate[1]-10, cate[0]))
# print('done')
# allimgs = os.listdir('data/New101/train/zzzzzzzz')
# for img in allimgs:
#     image = Image.open('data/New101/train/zzzzzzzz/'+img).convert('RGB')
#     image.save('data/New101/train/zzzzzzzz/'+img)


# allimgs = os.listdir('data/New101/val/zzzzzzzz')
# for img in allimgs:
#     image = Image.open('data/New101/val/zzzzzzzz/'+img).convert('RGB')
#     image.save('data/New101/val/zzzzzzzz/'+img)
allcates = os.listdir('data/New101/train')
for cate in allcates:
    # os.mkdir('data/New101/val/'+cate)
    tomove = random.sample(range(1, len(os.listdir('data/New101/train/'+cate))), 20)
    count=0
    for i in range(20):
        if count == 10:
            break
        try:
            shutil.move('data/New101/train/{}/image_{:0>4d}.jpg'.format(cate, tomove[i]), 'data/New101/val/{}/image_{:0>4d}.jpg'.format(cate, tomove[i]))
            count+=1
        except:
            print('no ')
# print('done')
# allfiles = os.listdir('data/New101/BACKGROUND_Google_bak')
# # print('total:{}'.format(467+len(allfiles)))
# count = 0
# for f in tqdm(allfiles):
#     if f.endswith('.jpg'):
#         try:
#             image = Image.open('data/New101/BACKGROUND_Google_bak/'+f)
#             count += 1
#             shutil.copy('data/New101/BACKGROUND_Google_bak/'+f, 'data/New101/BACKGROUND_Google/image_{:0>4d}.jpg'.format(count))
#         except:
#             print(f)


from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms.transforms import ToTensor

transform = transforms.Compose([
    transforms.RandomCrop(size=198, pad_if_needed=True),
    transforms.Resize(64),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(90),
    transforms.ToTensor()
])

train_dataset = ImageFolder('data/New101/train', transform=transform)
bg_sample = train_dataset.samples[-2012:]
bg_target = train_dataset.targets[-2012:]
train_dataset.samples += 10*bg_sample
train_dataset.targets += 10*bg_target

for i in tqdm(range(len(train_dataset))):
    train_dataset.__getitem__(i)

val_dataset = ImageFolder('data/New101/val')
for i in tqdm(range(len(val_dataset))):
    val_dataset.__getitem__(i)

# print('done')
# numbers = {}
# a = os.listdir('data/New101')
# for cate in a:
#     b = os.listdir('data/New101/'+cate)
#     numbers[cate]=len(b)
#     for img in b:
#         if not img.endswith('jpg'):
#             print(cate+'/'+img)

# with open('data/New101.json', 'w') as f:
#     f.write(json.dumps(numbers))
    # numbers.append()
# print(a)

# from torchvision.datasets import SVHN

# train_dataset = SVHN('data/SVHN/', split='train', download=True)
# test_dataset = SVHN('data/SVHN/', split='train', download=True)
# dataset = SVHN('data/SVHN/', split='train', download=True)



# class GradSeesawLoss(nn.Module):
#     def __init__(self, num_classes, p=0.8):
#         super(GradSeesawLoss, self).__init__()
#         self.num_classes = num_classes
#         cls_num_list = torch.ones(size=(num_classes,1)) * 1e-6
#         self.register_buffer('cls_num_list', cls_num_list)
#         self.p = p

        
#     @torch.no_grad()
#     def get_weight_matrix(self):
#         weight_matrix = (1.0 / self.cls_num_list) * self.cls_num_list.transpose(1,0)
#         weight_matrix[weight_matrix>1] = 1  
#         weight_matrix = torch.pow(weight_matrix, self.p)

#         weight_matrix = weight_matrix.cuda()
#         return weight_matrix

#     def forward(self, x, target):
#         '''
#         x: b * C
#         '''
#         bs = x.shape[0]
#         target_onehot = F.one_hot(target, num_classes=self.num_classes).float()
#         g = torch.abs(x.sigmoid().detach() - target_onehot)
#         # num_classes_batch = torch.sum(target_onehot*(1-g), 0, keepdim=True).detach().cpu().permute(1,0)
#         num_classes_batch = torch.sum(target_onehot, 0, keepdim=True).detach().cpu().permute(1,0)
#         self.cls_num_list += num_classes_batch
#         weight_matrix = self.get_weight_matrix()
#         weight = torch.mm(target_onehot, weight_matrix)
#         weighted_x = x + torch.log(weight + g)
#         softmax_x = F.softmax(weighted_x, 1)
        
#         loss = -torch.sum(target_onehot * torch.log(softmax_x)) / bs
#         return loss

# class SoftSeesawLoss(nn.Module):
#     def __init__(self, num_classes, p=0.8):
#         super(SoftSeesawLoss, self).__init__()
#         self.num_classes = num_classes
#         cls_num_list = torch.ones(size=(num_classes,1)) * 1e-6
#         self.register_buffer('cls_num_list', cls_num_list)
#         self.p = p

        
#     @torch.no_grad()
#     def get_weight_matrix(self):
#         weight_matrix = (1.0 / self.cls_num_list) * self.cls_num_list.transpose(1,0)
#         weight_matrix[weight_matrix>1] = 1  
#         weight_matrix = torch.pow(weight_matrix, self.p)

#         weight_matrix = weight_matrix.cuda()
#         return weight_matrix

#     def forward(self, x, target):
#         '''
#         x: b * C
#         '''
#         bs = x.shape[0]
#         target_onehot = F.one_hot(target, num_classes=self.num_classes).float()
#         # g = torch.abs(x.sigmoid().detach() - target_onehot)
#         confidence = F.softmax(x, 1).detach()
#         num_classes_batch = torch.sum(target_onehot*confidence, 0, keepdim=True).detach().cpu().permute(1,0)
#         self.cls_num_list += num_classes_batch
#         weight_matrix = self.get_weight_matrix()
#         weight = torch.mm(target_onehot, weight_matrix)
#         weighted_x = x + torch.log(weight)
#         softmax_x = F.softmax(weighted_x, 1)
        
#         loss = -torch.sum(target_onehot * torch.log(softmax_x))/bs
#         return loss
