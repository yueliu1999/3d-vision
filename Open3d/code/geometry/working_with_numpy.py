import open3d as o3d
import numpy as np

# generate numpy data
x = np.linspace(-3, 3, 401)
mesh_x, mesh_y = np.meshgrid(x, x)
z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
z_norm = (z - z.min()) / (z.max() - z.min())
xyz = np.zeros((np.size(mesh_x), 3))
xyz[:, 0] = np.reshape(mesh_x, -1)
xyz[:, 1] = np.reshape(mesh_y, -1)
xyz[:, 2] = np.reshape(z_norm, -1)
print('xyz')
print(xyz)

# from numpy to open3d.PointCloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
o3d.visualization.draw_geometries([pcd])

# from open3d.PointCloud to numpy
# Load saved point cloud and visualize it

# Convert Open3D.o3d.geometry.PointCloud to numpy array
xyz_load = np.asarray(pcd.points)
print('xyz_load')
print(xyz_load)