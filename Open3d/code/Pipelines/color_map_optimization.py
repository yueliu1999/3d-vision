import open3d as o3d
import numpy as np
import os

def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list_ordered, key=alphanum_key)


def get_file_list(path, extension=None):
    if extension is None:
        file_list = [
            path + f for f in os.listdir(path) if os.path.isfile(join(path, f))
        ]
    else:
        file_list = [
            path + f
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and
            os.path.splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)
    return file_list

path = o3dtut.download_fountain_dataset()
debug_mode = False

rgbd_images = []
depth_image_path = get_file_list(os.path.join(path, "depth/"), extension=".png")
color_image_path = get_file_list(os.path.join(path, "image/"), extension=".jpg")
assert (len(depth_image_path) == len(color_image_path))
for i in range(len(depth_image_path)):
    depth = o3d.io.read_image(os.path.join(depth_image_path[i]))
    color = o3d.io.read_image(os.path.join(color_image_path[i]))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False)
    if debug_mode:
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        o3d.visualization.draw_geometries([pcd])
    rgbd_images.append(rgbd_image)


