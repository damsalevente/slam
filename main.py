import cv2
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


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
        gt_imu_idx = t
        gps_idx = int(np.floor(0.01 * t))
        image_path = os.path.join(self.dataset_path, self._get_numpy('camera_data/color_left')[img_idx].decode('utf-8'))
        image_right = os.path.join(self.dataset_path, self._get_numpy('camera_data/color_right')[img_idx].decode('utf-8'))
        image_down = os.path.join(self.dataset_path, self._get_numpy('camera_data/color_down')[img_idx].decode('utf-8'))
        segmentation_path = os.path.join(self.dataset_path,self._get_numpy('camera_data/segmentation')[img_idx].decode('utf-8'))
        depth_path = os.path.join(self.dataset_path,self._get_numpy('camera_data/depth')[img_idx].decode('utf-8'))

        gyro = self._get_numpy('imu/gyroscope')[gt_imu_idx]
        accel = self._get_numpy('imu/accelerometer')[gt_imu_idx]
        position = self._get_numpy( 'groundtruth/position')[gt_imu_idx]
        velocity = self._get_numpy('groundtruth/velocity')[gt_imu_idx]

        gps = self._get_numpy('gps/position')[gps_idx] # floor (t)  in the midair data organization

        return (image_path,image_right,image_down,segmentation_path,depth_path, gyro, accel, position, velocity, gps)

def create_output(verts, colors, fname):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([verts.reshape(-1,3), colors])
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    with open(fname, "w") as f:
        f.write(ply_header % dict(vert_num = len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')

def process_3d_mesh(img1, img2):
    ''' 
    segfault for some reason, needs to be checked after training is finished 
    '''
    h, w = img1.shape[:2]
    print(h,w)
    img_left = cv2.pyrDown(img1)
    img_right =cv2.pyrDown(img2)
    win_size = 1
    min_disp = 32
    max_disp = min_disp * 9 
    num_disp = max_disp - min_disp
    print('process called')
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp, 
            numDisparities = num_disp,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 32,
            disp12MaxDiff = 1,
            P1 = 8 * 3 * win_size  ** 2,
            P2 = 32 * 3 * win_size ** 2
           )
    print(stereo)
    print('stereo created')
    disparity_map = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
    print("computing")
    h, w = img_left.shape[:2]
    focal_length = 0.8 * w 
    Q = np.float32([[1, 0, 0 , -w/2.0],
                    [0, -1, 0, h/2.0],
                    [0, 0, 0, -focal_length],
                    [0, 0, 1, 0]])
    points_3d = cv2.reprojectImageTo3D(disparity_map, Q)
    print('projection done')
    colors = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
    mask_map = disparity_map > disparity_map.min()
    output_points = points_3d[mask_map]
    output_colors = colors[mask_map]

    create_output(output_points, output_colors, 'result.ply')




if __name__ == "__main__":
    midair = MidAirDataset('/media/nap/rootMX18.1/home/levente/Dev/data/MidAir/PLE_training/fall')
    trajs = midair.get_trajectories()
    midair.select_trajectory_name('trajectory_4000')
    prev_img = None
    xdata = deque([100]*200)
    ydata = deque([5.0]*200)
    zdata = deque([4.0]*200)
    gt_xdata = deque([100]*200)
    gt_ydata = deque([5.0]*200)
    gt_zdata = deque([4.0]*200)
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection = '3d')

    for (idx, (img_left, img_right, img_down, seg, depth, gyro, accel, pos, velocity, gps_pos)) in enumerate(midair):
        if prev_img == img_left:
            continue
        prev_img = img_left
        img = cv2.imread(img_left, cv2.COLOR_BGR2HLS)
        img_r = cv2.imread(img_right, cv2.COLOR_BGR2HLS)
        print(img)
        print(img_r)

        img_depth = cv2.imread(depth, cv2.COLOR_BGR2HLS)
        img_depth_3ch = cv2.cvtColor(img_depth, cv2.COLOR_GRAY2RGB)
        img_conc = np.concatenate((img, img_depth_3ch), axis = 1)

        '''
        xdata.append(gps_pos[0])
        ydata.append(gps_pos[1])
        zdata.append(gps_pos[2])

        gt_xdata.append(pos[0])
        gt_ydata.append(pos[1])
        gt_zdata.append(pos[2])

        xdata.popleft()
        ydata.popleft()
        zdata.popleft()

        gt_xdata.popleft()
        gt_ydata.popleft()
        gt_zdata.popleft()

        ax.scatter3D(list(xdata), list(ydata), list(zdata), c = np.arange(200), cmap = 'Greens')
        ax.scatter3D(list(gt_xdata), list(gt_ydata), list(gt_zdata), c = np.arange(200), cmap = 'Blues')
        plt.draw()
        plt.pause(0.04)
        ax.cla()
        '''


        # test the 3d point cloud from stereo image from opencv example book
        process_3d_mesh(img, img_r)

        print("position: {}".format(pos))
        print("raw data: {} , {}".format(gyro, accel))

        cv2.imshow("image", img_conc)
        cv2.waitKey(1)

