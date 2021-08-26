import torch
from train import DeltaDataset
from model import darknet53
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

data = torch.load('./result_20.pth')

test_loader = DataLoader(DeltaDataset(), batch_size = 1) 
print('loaded')
model = darknet53(7).to('cuda')
model.load_state_dict(data['state_dict'])
model.eval()
print('model_loaded')
abs_pos = np.zeros(3)
gt_abs_pos = np.zeros(3)
fig = plt.figure()
ax = plt.axes(projection='3d')
i = 0
arr_abs_pos = []
arr_gt_abs_pos = []
with torch.no_grad():
    for idx, data in enumerate(test_loader, 0):
        img = data[0].to('cuda')
        pos = data[1].to('cuda')
        label_c = data[2].to('cuda')
        output = model(img, pos)
        abs_pos = abs_pos + output[0][:3].to('cpu').numpy()
        gt_abs_pos= gt_abs_pos + label_c[0][:3].to('cpu').numpy()
        print(output[0][:3].to('cpu').numpy())
        arr_abs_pos.append(abs_pos)
        arr_gt_abs_pos.append(gt_abs_pos)
        i += 1
        if i == 1000:
            break
arr_abs_pos = np.array(arr_abs_pos)
arr_gt_abs_pos = np.array(arr_gt_abs_pos)
ax.scatter(arr_abs_pos[:,0], arr_abs_pos[:,1], arr_abs_pos[:,2], cmap = 'viridis')
ax.scatter(arr_gt_abs_pos[:,0], arr_gt_abs_pos[:,1], arr_gt_abs_pos[:,2], cmap = 'cividis')
plt.show()
