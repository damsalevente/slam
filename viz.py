import open3d as o3d
import numpy as np 

point_cloud = o3d.io.read_point_cloud('./result.ply')
print(point_cloud)

o3d.visualization.draw_geometries([point_cloud])

