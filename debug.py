from losses import SeesawLoss
cls_num_list = []
for i in range(100, 0, -1):
    cls_num_list.append(i)

loss = SeesawLoss(cls_num_list)

print('done')