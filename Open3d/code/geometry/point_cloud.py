import open3d as o3d
import numpy as np
# visualize point cloud
pcd = o3d.io.read_point_cloud("./test_data/input.ply")
print(pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd])

# voxel downsampling
downpcd = pcd.voxel_down_sample(voxel_size = 0.5)
o3d.visualization.draw_geometries([downpcd])

# vertex normal estimation
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
o3d.visualization.draw_geometries([downpcd], point_show_normal=True)

# access estimated vertex normal
print(downpcd.normals[0])
print(np.asarray(downpcd.normals)[:10, :])

# crop point cloud
pass

# paint point cloud
downpcd.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([downpcd])

# point cloud distance
pass

# bounding volumes
aabb = pcd.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)
obb = pcd.get_oriented_bounding_box()
obb.color = (0, 1, 0)
o3d.visualization.draw_geometries([pcd, aabb, obb])

# convex hull
