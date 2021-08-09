import cv2
import os
import h5py
import numpy as np

from PIL import Image


class MidAirDataset:
    def __init__(self, dataset_path = '/media/nap/rootMX18.1/home/levente/Dev/data/MidAir/PLE_training/fall'):
        # hard code first, think second
        self.dataset_path = dataset_path
        self.color_left_path = os.path.join(dataset_path, 'color_left')
        self.depth_path = os.path.join(dataset_path, 'depth')
        self.seg_path = os.path.join(dataset_path, 'segmentation')
        self.ann_file = os.path.join(dataset_path, 'sensor_records.hdf5')
        self.f1 = h5py.File(self.ann_file)
        #self.dset = f1['trajectory_4006']
        self.trajectories = self.get_trajectories()
        self.selected_trajectory = self.trajectories[0]
        # N is the trajectory length in seconds 
        print("init done")

    def get_trajectories(self):
        return list(self.f1.keys())

    def select_trajectory_idx(self, idx):
        self.selected_trajectory = self.trajectories[idx]

    def select_trajectory_name(self, name):
        self.selected_trajectory = name

    def _get_numpy(self, what):
        return np.array(self.f1[self.selected_trajectory].get(what))

    def _trajectory_len(self):
        return len(self.trajectories)

    def __len__(self):
        '''
        return current trajectory lenght in seconds
        '''
        traj_len = self._get_numpy('camera_data/color_left').shape[0]
        print(traj_len)
        N =  int(traj_len // 25)
        return N * 100 # true index 

    def __getitem__(self, t):
        '''
        return for time t in ms 
        t could be float!!
        return image, image path, segmentation path, gyros, accel, and other ground truth values for slam
        '''
        #color_down  color_left  color_right  depth  segmentation
        img_idx = int(np.floor(0.25 * t))
        gt_imu_idx = int(np.floor(t))
        image_path = os.path.join(self.dataset_path,self._get_numpy('camera_data/color_left')[img_idx])
        image_right = os.path.join(self.dataset_path,self._get_numpy('camera_data/color_right')[img_idx])
        image_down = os.path.join(self.dataset_path, self._get_numpy('camera_data/color_down')[img_idx])
        segmentation_path = os.path.join(self.dataset_path,self._get_numpy('camera_data/segmentation')[img_idx])
        depth_path = os.path.join(self.dataset_path,self._get_numpy('camera_data/depth')[img_idx])

        gyro = self._get_numpy('imu/gyroscope')[gt_imu_idx]
        accel = self._get_numpy('imu/accelerometer')[gt_imu_idx]
        position = self._get_numpy( 'groundtruth/position')[gt_imu_idx]
        velocity = self._get_numpy('groundtruth/velocity')[gt_imu_idx]

        return (image_path,image_right,image_down,segmentation_path,depth_path, gyro, accel, position, velocity)

if __name__ == "__main__":
    midair = MidAirDataset('/media/nap/rootMX18.1/home/levente/Dev/data/MidAir/PLE_training/fall') # default path used 
    trajs = midair.get_trajectories()
    midair.select_trajectory_name('trajectory_4000')
    prev_img = None
    for (img_left, img_right, img_down, seg, depth, gyro, accel, pos, velocity) in midair:
        if prev_img == img_left:
            continue
        prev_img = img_left
        img = cv2.imread(img_left, cv2.COLOR_BGR2HLS)
        img_depth = cv2.imread(depth, cv2.COLOR_BGR2HLS)
        img_depth_3ch = cv2.cvtColor(img_depth, cv2.COLOR_GRAY2RGB)
        img_conc = np.concatenate((img, img_depth_3ch), axis = 1)
        print("position: {}".format(pos))
        print("raw data: {} , {}".format(gyro, accel))
        cv2.imshow("image", img_conc)
        cv2.waitKey(1)
