
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torchvision import transforms
# from torch.utils.data import Dataloader
import torch
import numpy as np
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from PIL import Image


a = ImageFolder('data/TinyImageNet_bg/train')
print(a)
# transform_val = transforms.Compose([
#         transforms.Resize(224),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])

# dataset = ImageFolder('data/TinyImageNet/val', transform=transform_val)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, pin_memory=True)

# model = resnet50(pretrained=True)
# model.cuda()
# model.eval()

# cls_result = []

# with torch.no_grad():
#     for (input, target) in tqdm(dataloader):
#         input = input.cuda()
#         target = target.cuda()
#         output = model(input)
#         predict = torch.argmax(output, 1).cpu().numpy()
#         # predict = 
#         cls_result += list(predict)

# result = np.asarray(cls_result)
# np.save('cls_result.npy', result)


cls_result = list(np.load('cls_result.npy'))
set_temp=set(cls_result)
dict_temp={}
for item in set_temp:
    dict_temp.update({item:cls_result.count(item)})
temp = []
for key in dict_temp.keys():
    temp.append([key, dict_temp[key]])
temp.sort(key=lambda x: x[1], reverse=True)
temp = temp[:200]
print(dict)