import matplotlib.pyplot as plt 
from nuscenes.nuscenes import NuScenes
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
import numpy as np
import cv2
from torchvision.io import read_image
import torch
from pyquaternion import Quaternion

class NusDeltaDataset(Dataset):
    def __init__(self, dataset_path = '/media/nap/rootMX18.1/home/levente/Dev/data/nuscenes/', transforms = None):
        self.dataset_path = dataset_path
        self.nusc = NuScenes(version = 'v1.0-mini', dataroot = self.dataset_path, verbose = True)
        self.data = []
        self.count = 0
        for element in nusc.scene:
            # for every sample in that scene
            print(element)
            token = element['first_sample_token']
            while token != "":
                # get the current sample 
                sample = nusc.get('sample', token)
                # get the ego pose and img front 
                camera_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
                image = os.path.join(dataset_path,camera_data['filename'])
                pose = nusc.get('ego_pose',camera_data['ego_pose_token'])
                #print(camera_data)
                #print(pose)
                self.data.append(np.array([element['name'],image, pose]))
                self.count += 1
                token = sample['next']
        self.nusc = None # ? 
        self.transforms = transforms



    def __len__(self):
        '''
        return current trajectory length in 10ms 
        '''
        return self.count - 1

    def handle_image(self, img_path):
        image = read_image(img_path)
        if self.transforms:
            image = self.transforms(image)
        return image


    def __getitem__(self, idx):
        # handle new scene case
        element = self.data[idx]
        future = self.data[idx+1]
        if element[0] != future[0]: # name mismatch, e.g new scene
            future = element # do the previous one 
            element = self.data[idx-1]
        # load the images 
        img = self.handle_image(element[1])
        img_next = self.handle_image(future[1])

        gt_translation = element[2]['translation']
        gt_translation_next = future[2]['translation']

        gt_rotation = Quaternion(element[2]['rotation']).normalised
        gt_rotation_next = Quaternion(future[2]['rotation']).normalised

        translation = torch.tensor(gt_translation_next) - torch.tensor(gt_translation)
        rotation = torch.tensor((gt_rotation.conjugate * gt_rotation_next).elements) # start position inverse * actual quatertion yields the difference between them
        print(translation)
        print(rotation)

        return torch.cat([img, img_next]), torch.cat([translation, rotation]).flatten()


if __name__ == "__main__":
    dataset_path = '/media/nap/rootMX18.1/home/levente/Dev/data/nuscenes/'
    nusc = NuScenes(version = 'v1.0-mini', dataroot = dataset_path, verbose = True)
    ds = NusDeltaDataset()
