import torch
from train import DeltaDataset
from model import darknet53
from torch.utils.data import Dataset, DataLoader

data = torch.load('./result_20.pth')

test_loader = DataLoader(DeltaDataset(), batch_size = 1) 
print('loaded')
model = darknet53(7).to('cuda')
model.load_state_dict(data['state_dict'])
model.eval()
print('model_loaded')
abs_pos = torch.zeros(7)
with torch.no_grad():
    for idx, data in enumerate(test_loader, 0):
        img = data[0].to('cuda')
        pos = data[1].to('cuda')
        label_c = data[2].to('cuda')
        output = model(img, pos)
        print(output.shape)
        abs_pos = torch.add(abs_pos,output.to('cpu'))
        print('{}'.format(abs_pos))
        print(data[3])
