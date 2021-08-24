from model import darknet53
from main import MidAirDataset
import numpy as np 
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import quaternion

def loss_function(y_true, y):
    trasnlation_loss = torch.nn.functional.mse_loss(y_true[:,:3], y[:,:3])
    angle = torch.nn.functional.mse_loss(y_true[:,3:], y[:,3:])
    loss = angle + trasnlation_loss
    return loss

trans = transforms.Compose([
    transforms.ToTensor()
    ])
class DeltaDataset(Dataset):
    def __init__(self):
        self.midair = MidAirDataset('/media/nap/rootMX18.1/home/levente/Dev/data/MidAir/PLE_training/fall')
        self.midair.select_trajectory_name('trajectory_4000')

    def __len__(self):
        return len(self.midair) - 8

    def __getitem__(self,idx):
        # X 
        img = cv2.imread(self.midair[idx].image_left, cv2.COLOR_BGR2HLS)
        data = [self.midair[idx].gyro]
        data.append(self.midair[idx].accel)
        for j in range(1,8):
            data.append(self.midair[idx + j].gyro)
            data.append(self.midair[idx + j].accel)
        dd_future = self.midair[idx + 7] # + 10 ms
        img_next = cv2.imread(dd_future.image_left,cv2.COLOR_BGR2HLS)
        X = np.concatenate((img, img_next))
        # y 
        y_pos = dd_future.pos - self.midair[idx].pos
        # https://stackoverflow.com/questions/1755631/difference-between-two-quaternions
        y_angle = quaternion.as_float_array(dd_future.to_quart() * np.conjugate(self.midair[idx].to_quart()))
        y = np.append(y_pos, y_angle)

        return trans(X), torch.tensor(data, dtype=torch.float32).flatten(),  torch.tensor(y, dtype=torch.float32).flatten()

def save_model(model, optmizer, epoch):
    state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optmizer.state_dict()
    }
    torch.save(state, "result_{}.pth".format(epoch))

if __name__ == "__main__":
    device = 'cuda'
    model = darknet53(7).to(device)

    trainloader = DataLoader(DeltaDataset(), batch_size = 8, shuffle = False, pin_memory = True) 

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)

    for epoch in range(40):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            img, data, labels = data
            img = img.to(device)
            data = data.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(img,data)
            loss = loss_function(labels, outputs)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        save_model(model,optimizer, epoch+1)

    print('Finished Training')
