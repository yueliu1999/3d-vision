import open3d as o3d
import numpy as np

# face reconstruction
pcd = o3d.io.read_point_cloud("../test_data/test.ply")
o3d.visualization.draw_geometries([pcd])

# alpha shapes
alpha = 0.1
print(f"alpha={alpha:.3f}")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
# 遍历多个alpha
for alpha in np.logspace(np.log10(0.5), np.log10(0.01), num=4):
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha, tetra_mesh, pt_map)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# ball pivoting
radii = [0.5, 0.1, 0.2, 0.4]
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))
o3d.visualization.draw_geometries([pcd, rec_mesh])

# poisson surface reconstruction

# norm estimate
pcd.orient_normals_consistent_tangent_plane(100)
o3d.visualization.draw_geometries([pcd], point_show_normal=True)