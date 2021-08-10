import cv2
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import struct
from scipy.spatial.transform import Rotation 


from PIL import Image

DRAW_PCL = os.getenv("DRAW_PCL") if not None else None # whether to create a point cloud file ( it is slow ) 
DRAW_PCL_STEREO = os.getenv("DRAW_PCL_STEREO") if not None else None # same as above, but with the stereo camera method
DRAW_SCATTER = os.getenv("DRAW_SCATTER") if not None else None # draw a 3d scatter plot based on images
DRAW_IMG = os.getenv("DRAW_IMG") if not None else None

class DroneData:
    def __init__(self, image_path,image_right,image_down,segmentation_path,depth_path, gyro, accel, position, velocity, gps, ang_vel, attittude):
        self.image_left = image_path
        self.image_right = image_right
        self.image_down = image_down
        self.segmentation_path = segmentation_path
        self.depth_path = depth_path
        self.gyro = self._ned2xyz(gyro)
        self.accel = self._ned2xyz(accel)
        self.pos = self._ned2xyz(position)
        self.velocity = self._ned2xyz(velocity)
        self.gps = gps
        self.angular_velocity = ang_vel
        self.attitude = attittude

    def _ned2xyz(self, data):
        return np.array([data[0], -data[1], -data[2]])

    def calc_6dof(self):
        '''
        ground truth could be either the querterinos or the euler angles 
        for now i try to calculate the euler matrix from the gt querterinos
        '''
        print('ATT: {}'.format(self.attitude))
        q = self.attitude #don't want to write too much 
        position = self.pos
        # normalize 
        R =  2 * np.array([ [q[0] * q[0] + q[1] * q[1] - 0.5, q[1] * q[2] - q[0]*q[3], q[1]*q[3] + q[0]*q[2]],
                          [q[1] * q[2] + q[0]*q[3], q[0] **2 + q[2] ** 2 - 0.5, q[2]*q[3] - q[0] * q[1]],
                          [q[1] * q[3] - q[0]*q[2], q[2]*q[3] + q[0] * q[3], q[0]**2 + q[3] **2 - 0.5]])

        sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        singular = sy  < 1e-6
        if not singular:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
        return position, np.array([x,y,z])

    def calc_euler(self):
        rot = Rotation.from_quat(self.attitude)
        rot_euler = rot.as_euler('xyz',degrees = False)
        return rot_euler

        
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
        ang_velocity = self._get_numpy('groundtruth/angular_velocity')[gt_imu_idx]
        att = self._get_numpy('groundtruth/attitude')[gt_imu_idx]

        gps = self._get_numpy('gps/position')[gps_idx] # floor (t)  in the midair data organization

        return DroneData(image_path,image_right,image_down,segmentation_path,depth_path, gyro, accel, position, velocity, gps, ang_velocity, att)
        

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

def get_pointcloud(color_image, depth_image, camera_intrinsics):
    """ from: https://gist.github.com/Shreeyak/9a4948891541cb32b501d058db227fff
    creates 3D point cloud of rgb images by taking depth information
        input : color image: numpy array[h,w,c], dtype= uint8
                depth image: numpy array[h,w] values of all channels will be same
        output : camera_points, color_points - both of shape(no. of pixels, 3)
    """
    color_image = cv2.pyrDown(color_image)
    depth_image = cv2.pyrDown(depth_image)
    image_height = depth_image.shape[0]
    image_width = depth_image.shape[1]
    pixel_x,pixel_y = np.meshgrid(np.linspace(0,image_width-1,image_width),
                                  np.linspace(0,image_height-1,image_height))
    camera_points_x = np.multiply(pixel_x-camera_intrinsics[0,2],depth_image/camera_intrinsics[0,0])
    camera_points_y = np.multiply(pixel_y-camera_intrinsics[1,2],depth_image/camera_intrinsics[1,1])
    camera_points_z = depth_image
    camera_points = np.array([camera_points_x,camera_points_y,camera_points_z]).transpose(1,2,0).reshape(-1,3)

    color_points = color_image.reshape(-1,3)

    # Remove invalid 3D points (where depth == 0)
    valid_depth_ind = np.where(depth_image.flatten() > 0)[0]
    camera_points = camera_points[valid_depth_ind,:]
    color_points = color_points[valid_depth_ind,:]

    return camera_points,color_points

def write_pointcloud(filename,xyz_points,rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                        rgb_points[i,2].tostring())))
    fid.close()

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
    # midair dataset stuff
    midair = MidAirDataset('/media/nap/rootMX18.1/home/levente/Dev/data/MidAir/PLE_training/fall')
    trajs = midair.get_trajectories()
    midair.select_trajectory_name('trajectory_4000')
    prev_img = None # for main loop to skip same images to redraw
    # imu plotting stuff
    dbuf_size = 10000
    xdata = deque([100]*dbuf_size)
    ydata = deque([5.0]*dbuf_size)
    zdata = deque([4.0]*dbuf_size)
    gt_xdata = deque([100]*dbuf_size)
    gt_ydata = deque([5.0]*dbuf_size)
    gt_zdata = deque([4.0]*dbuf_size)
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    # rgbd stuff 
    fx = 512 # in pixels, fov = 90 degree, image size = 1024: focal length: (image_size/2) / tg(fov/2)
    fy = 512 # same size
    cx = 256
    cy = 256

    camera_intrinsics  = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    fname = "rgbd_pointcloud.ply"
    for idx, drone_data in enumerate(midair):
        pos, angle = drone_data.calc_6dof()
        angle2 = drone_data.calc_euler()
        print('angle: {}\nrepo: {}\nDiff:{}\n'.format(angle, angle2, angle2  - angle))
        exit(1)
        if idx % 4 != 0:
            continue
        prev_img = drone_data.image_left
        img = cv2.imread(drone_data.image_left, cv2.COLOR_BGR2HLS)
        img_r = cv2.imread(drone_data.image_right, cv2.COLOR_BGR2HLS)

        #img_depth = np.stack([cv2.imread(drone_data.depth_path, cv2.COLOR_BGR2HLS)] * 3, axis = 2)
        #print(img_depth.shape)
        #print(img.shape)
        img_conc = np.concatenate((img, img_r))

        xdata.append(drone_data.pos[0])
        ydata.append(drone_data.pos[1])
        zdata.append(drone_data.pos[2])

        #gt_xdata.append(drone_data.pos[0])
        #gt_ydata.append(drone_data.pos[1])
        #gt_zdata.append(drone_data.pos[2])

        xdata.popleft()
        ydata.popleft()
        zdata.popleft()

        #gt_xdata.popleft()
        #gt_ydata.popleft()
        #gt_zdata.popleft()
        if DRAW_SCATTER:
            ax.scatter3D(list(xdata), list(ydata), list(zdata), c = np.arange(dbuf_size), cmap = 'Greens')
            plt.draw()
            plt.pause(0.02)
            ax.cla()


        # test the 3d point cloud from stereo image from opencv example book
        if DRAW_PCL_STEREO:
            process_3d_mesh(img, img_r) 

        # test the rgbd image 
        if DRAW_PCL:
            c_points, color_points = get_pointcloud(img, img_depth, camera_intrinsics)
            write_pointcloud(fname, c_points, color_points)

        if DRAW_IMG:
            cv2.imshow("image", img_conc)
        cv2.waitKey(1)

