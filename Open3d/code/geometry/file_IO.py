import open3d as o3d

print("Testing IO for point cloud ...")
pcd = o3d.io.read_point_cloud("../test_data/input.ply")
print(pcd)
o3d.io.write_point_cloud("../test_data/input.ply", pcd)

print("Testing IO for meshes ...")
mesh = o3d.io.read_triangle_mesh("../test_data/knot.ply")
print(mesh)
o3d.io.write_triangle_mesh("../test_data/knot.ply", mesh)

print("Testing IO for images ...")
img = o3d.io.read_image("../test_data/color/00000.jpg")
print(img)
o3d.io.write_image("../test_data/color/00000.jpg", img)