import numpy as np
import open3d as o3d
import copy
# read the ply file
print("Testing mesh in Open3D...")
mesh = o3d.io.read_triangle_mesh("../test_data/knot.ply")
print(mesh)
print('Vertices:')
print(np.asarray(mesh.vertices))
print('Triangles:')
print(np.asarray(mesh.triangles))

# visualize a 3d mesh
print("Try to render a mesh with normals (exist: " +
      str(mesh.has_vertex_normals()) + ") and colors (exist: " +
      str(mesh.has_vertex_colors()) + ")")
o3d.visualization.draw_geometries([mesh])
print("A mesh with no normals and no colors does not look good.")

# surface normal estimate
print("Computing normal and rendering it.")
mesh.compute_vertex_normals()
print(np.asarray(mesh.triangle_normals))
print("Try to render a mesh with normals (exist: " +
      str(mesh.has_vertex_normals()) + ") and colors (exist: " +
      str(mesh.has_vertex_colors()) + ")")
o3d.visualization.draw_geometries([mesh])

# crop mesh
print("We make a partial mesh of only the first half triangles.")
mesh1 = copy.deepcopy(mesh)
mesh1.triangles = o3d.utility.Vector3iVector(
    np.asarray(mesh1.triangles)[:len(mesh1.triangles) // 2, :])
mesh1.triangle_normals = o3d.utility.Vector3dVector(
    np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals) // 2, :])
print(mesh1.triangles)
o3d.visualization.draw_geometries([mesh1])

# paint mesh
print("Painting the mesh")
mesh1.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([mesh1])

# Mesh properties
# def check_properties(name, mesh):
#     mesh.compute_vertex_normals()
#
#     edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
#     edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
#     vertex_manifold = mesh.is_vertex_manifold()
#     self_intersecting = mesh.is_self_intersecting()
#     watertight = mesh.is_watertight()
#     orientable = mesh.is_orientable()
#
#     print(name)
#     print(f"  edge_manifold:          {edge_manifold}")
#     print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
#     print(f"  vertex_manifold:        {vertex_manifold}")
#     print(f"  self_intersecting:      {self_intersecting}")
#     print(f"  watertight:             {watertight}")
#     print(f"  orientable:             {orientable}")
#
#     geoms = [mesh]
#     if not edge_manifold:
#         edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
#         geoms.append(o3dtut.edges_to_lineset(mesh, edges, (1, 0, 0)))
#     if not edge_manifold_boundary:
#         edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
#         geoms.append(o3dtut.edges_to_lineset(mesh, edges, (0, 1, 0)))
#     if not vertex_manifold:
#         verts = np.asarray(mesh.get_non_manifold_vertices())
#         pcl = o3d.geometry.PointCloud(
#             points=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[verts]))
#         pcl.paint_uniform_color((0, 0, 1))
#         geoms.append(pcl)
#     if self_intersecting:
#         intersecting_triangles = np.asarray(
#             mesh.get_self_intersecting_triangles())
#         intersecting_triangles = intersecting_triangles[0:1]
#         intersecting_triangles = np.unique(intersecting_triangles)
#         print("  # visualize self-intersecting triangles")
#         triangles = np.asarray(mesh.triangles)[intersecting_triangles]
#         edges = [
#             np.vstack((triangles[:, i], triangles[:, j]))
#             for i, j in [(0, 1), (1, 2), (2, 0)]
#         ]
#         edges = np.hstack(edges).T
#         edges = o3d.utility.Vector2iVector(edges)
#         geoms.append(o3dtut.edges_to_lineset(mesh, edges, (1, 0, 1)))
#     o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)

# mesh filter
# average filter
vertices = np.asarray(mesh.vertices)
noise = 5
vertices += np.random.uniform(0, noise, size=vertices.shape)
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])

print('filter with average with 1 iteration')
mesh_out = mesh.filter_smooth_simple(number_of_iterations=1)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out])

print('filter with average with 5 iterations')
mesh_out = mesh.filter_smooth_simple(number_of_iterations=5)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out])

# laplace filter
print('filter with Laplacian with 10 iterations')
mesh_out = mesh.filter_smooth_laplacian(number_of_iterations=10)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out])

print('filter with Laplacian with 50 iterations')
mesh_out = mesh.filter_smooth_laplacian(number_of_iterations=50)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out])

# taubin filter
mesh_out = mesh.filter_smooth_taubin(number_of_iterations=10)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out])

print('filter with Taubin with 100 iterations')
mesh_out = mesh.filter_smooth_taubin(number_of_iterations=100)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out])

# sampling
mesh = o3d.geometry.TriangleMesh.create_sphere()
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
pcd = mesh.sample_points_uniformly(number_of_points=500)
o3d.visualization.draw_geometries([pcd])

# mesh subdivision
mesh = o3d.geometry.TriangleMesh.create_box()
mesh.compute_vertex_normals()
print(
    f'The mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
)
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
mesh = mesh.subdivide_midpoint(number_of_iterations=1)
print(
    f'After subdivision it has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
)
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

# mesh subdivision
mesh = o3d.geometry.TriangleMesh.create_sphere()
mesh.compute_vertex_normals()
print(
    f'The mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
)
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
mesh = mesh.subdivide_loop(number_of_iterations=2)
print(
    f'After subdivision it has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
)
o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)

# Mesh simplification
print(
    f'Input mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
)
o3d.visualization.draw_geometries([mesh])

voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 32
print(f'voxel_size = {voxel_size:e}')
mesh_smp = mesh.simplify_vertex_clustering(
    voxel_size=voxel_size,
    contraction=o3d.geometry.SimplificationContraction.Average)
print(
    f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
)
o3d.visualization.draw_geometries([mesh_smp])

voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 16
print(f'voxel_size = {voxel_size:e}')
mesh_smp = mesh.simplify_vertex_clustering(
    voxel_size=voxel_size,
    contraction=o3d.geometry.SimplificationContraction.Average)
print(
    f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
)
o3d.visualization.draw_geometries([mesh_smp])

# connected components
print("Generate data")
vert = np.asarray(mesh.vertices)
min_vert, max_vert = vert.min(axis=0), vert.max(axis=0)
for _ in range(30):
    cube = o3d.geometry.TriangleMesh.create_box()
    cube.scale(0.005, center=cube.get_center())
    # 对cube进行平移
    cube.translate(
        (
            np.random.uniform(min_vert[0], max_vert[0]),
            np.random.uniform(min_vert[1], max_vert[1]),
            np.random.uniform(min_vert[2], max_vert[2]),
        ),
        relative=False,
    )
    mesh += cube
mesh.compute_vertex_normals()
print("Show input mesh")
o3d.visualization.draw_geometries([mesh])

print("Cluster connected triangles")
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    triangle_clusters, cluster_n_triangles, cluster_area = (
        mesh.cluster_connected_triangles())
triangle_clusters = np.asarray(triangle_clusters)
cluster_n_triangles = np.asarray(cluster_n_triangles)
cluster_area = np.asarray(cluster_area)

print("Show mesh with small clusters removed")
mesh_0 = copy.deepcopy(mesh)
triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
mesh_0.remove_triangles_by_mask(triangles_to_remove)
o3d.visualization.draw_geometries([mesh_0])

print("Show largest cluster")
mesh_1 = copy.deepcopy(mesh)
largest_cluster_idx = cluster_n_triangles.argmax()
triangles_to_remove = triangle_clusters != largest_cluster_idx
mesh_1.remove_triangles_by_mask(triangles_to_remove)
o3d.visualization.draw_geometries([mesh_1])



