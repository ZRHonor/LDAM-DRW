from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import numpy as np


class ImTinyImagenet(ImageFolder):
    def __init__(self,
                root: str,
                imb_type='exp', 
                imb_factor=0.01,
                rand_number=0,
                train=True,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                loader: Callable[[str], Any] = default_loader,
                is_valid_file: Optional[Callable[[str], bool]] = None):
        super().__init__(root, transform=transform, target_transform=target_transform, loader=loader, is_valid_file=is_valid_file)
        if not train:
            imb_factor = 1
        np.random.seed(rand_number)
        self.cls_num = len(self.classes)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)


        # print('done')

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.samples) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_samples = []
        new_targets = []
        new_imgs = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_samples.extend([self.samples[i] for i in selec_idx])
            new_imgs.extend([self.imgs[i] for i in selec_idx])
            # new_samples.append(self.samples[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        # new_samples = np.vstack(new_samples)
        self.samples = new_samples
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
    

    # def __getitem__(self, index: int) -> Tuple[Any, Any]:
    #     sample, target = super().__getitem__(index)
    #     # target = self.class_to_idx[target]
    #     print('done')
    #     return 
class ImTinyImagenet_bg(ImTinyImagenet):
    def __init__(self, root: str, imb_type, imb_factor, rand_number, train,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                loader: Callable[[str], Any] = default_loader,
                is_valid_file: Optional[Callable[[str], bool]] = None):
        super().__init__(root, imb_type=imb_type, imb_factor=imb_factor, rand_number=rand_number, train=train, transform=transform, target_transform=target_transform, loader=loader, is_valid_file=is_valid_file)
        

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        # img_max = len(self.samples) / cls_num
        img_max = 1000
        cls_num = 100
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        bg_num= min(70000, int(0.1*np.asarray(img_num_per_cls).sum()))
        img_num_per_cls.append(bg_num)
        return img_num_per_cls 

# class ImTinyImagenet_bg(ImageFolder):
#     def __init__(self,
#             root: str,
#             imb_type='exp', 
#             imb_factor=0.01,
#             rand_number=0,
#             train=True,
#             transform: Optional[Callable] = None,
#             target_transform: Optional[Callable] = None,
#             loader: Callable[[str], Any] = default_loader,
#             is_valid_file: Optional[Callable[[str], bool]] = None):
#         super().__init__(root, transform=transform, target_transform=target_transform, loader=loader, is_valid_file=is_valid_file)
#         if not train:
#             imb_factor = 1
#         np.random.seed(rand_number)
#         self.cls_num = len(self.classes) -1
#         img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
#         self.gen_imbalanced_data(img_num_list)


#         # print('done')

#     def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
#         # img_max = len(self.samples) / cls_num
#         img_max = 1000
#         img_num_per_cls = []
#         if imb_type == 'exp':
#             for cls_idx in range(cls_num):
#                 num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
#                 img_num_per_cls.append(int(num))
#         elif imb_type == 'step':
#             for cls_idx in range(cls_num // 2):
#                 img_num_per_cls.append(int(img_max))
#             for cls_idx in range(cls_num // 2):
#                 img_num_per_cls.append(int(img_max * imb_factor))
#         else:
#             img_num_per_cls.extend([int(img_max)] * cls_num)
#         return img_num_per_cls

#     def gen_imbalanced_data(self, img_num_per_cls):
#         new_samples = []
#         new_targets = []
#         new_imgs = []
#         targets_np = np.array(self.targets[:-1], dtype=np.int64)
#         classes = np.unique(targets_np)
#         # np.random.shuffle(classes)
#         self.num_per_cls_dict = dict()
#         for the_class, the_img_num in zip(classes, img_num_per_cls):
#             self.num_per_cls_dict[the_class] = the_img_num
#             idx = np.where(targets_np == the_class)[0]
#             np.random.shuffle(idx)
#             selec_idx = idx[:the_img_num]
#             new_samples.extend([self.samples[i] for i in selec_idx])
#             new_imgs.extend([self.imgs[i] for i in selec_idx])
#             # new_samples.append(self.samples[selec_idx, ...])
#             new_targets.extend([the_class, ] * the_img_num)
#         # new_samples = np.vstack(new_samples)
#         self.samples = new_samples
#         self.targets = new_targets

#     def get_cls_num_list(self):
#         cls_num_list = []
#         for i in range(self.cls_num):
#             cls_num_list.append(self.num_per_cls_dict[i])
#         return cls_num_list



if __name__ == "__main__":
    a = ImTinyImagenet(root='data/TinyImageNet/train')
    a.__getitem__(0)
    print('done')