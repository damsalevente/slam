import torch
from train import DeltaDataset
from model import darknet53
from torch.utils.data import Dataset, DataLoader

data = torch.load('./result_3.pth')

ds = DeltaDataset()
print('loaded')
test_loader = DataLoader(DeltaDataset(), batch_size = 1) 
print('loaded')
model = darknet53(7).to('cuda')
model.load_state_dict(data['state_dict'])
print('model_loaded')

with torch.no_grad():
    for idx, data in enumerate(test_loader, 0):
        img = data[0].to('cuda')
        pos = data[1].to('cuda')
        label_c = data[2].to('cuda')
        output = model(img, label_c)
        print('label: {}\n prediction: {}\n'.format(data[2], output.to('cpu')))
        print('diff: {}'.format(data[2] - output.to('cpu')))
