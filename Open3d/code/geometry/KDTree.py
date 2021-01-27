import open3d as o3d
import numpy as np
# built KDTree from point cloud
print("Testing kdtree in Open3D...")
print("Load a point cloud and paint it gray.")
pcd = o3d.io.read_point_cloud("../test_data/knot.ply")
pcd.paint_uniform_color([0.5, 0.5, 0.5])
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

print("Paint the 50th point red.")
pcd.colors[50] = [1, 0, 0]

print("Find its 200 nearest neighbors, and paint them blue.")
[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[50], 200)
np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]

[k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[50], 20)
np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]

print("Visualize the point cloud.")
o3d.visualization.draw_geometries([pcd])