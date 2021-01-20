## Geometry

### Basic

- Point cloud

  - Visualize point cloud

    ```python
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud("../../test_data/fragment.ply")
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
    ```

    **read_point_cloud** 可以在文件中读入点云，自动处理后缀

    支持的后缀：

    - xyz
    - xyzn
    - xyzrgb
    - pts
    - ply
    - pcd

    可以明确指定格式，这个时候后缀将会被无视：

    ```python
    pcd = o3d.io.read_point_cloud("../../test_data/my_points.txt", format='xyz')
    ```

    **draw_geometries** 将点云可视化 

    可以用h来显示help

    ```python
    [Open3D INFO]   -- Mouse view control --
    [Open3D INFO]     Left button + drag         : Rotate.
    [Open3D INFO]     Ctrl + left button + drag  : Translate.
    [Open3D INFO]     Wheel button + drag        : Translate.
    [Open3D INFO]     Shift + left button + drag : Roll.
    [Open3D INFO]     Wheel                      : Zoom in/out.
    [Open3D INFO] 
    [Open3D INFO]   -- Keyboard view control --
    [Open3D INFO]     [/]          : Increase/decrease field of view.
    [Open3D INFO]     R            : Reset view point.
    [Open3D INFO]     Ctrl/Cmd + C : Copy current view status into the clipboard.
    [Open3D INFO]     Ctrl/Cmd + V : Paste view status from clipboard.
    [Open3D INFO] 
    [Open3D INFO]   -- General control --
    [Open3D INFO]     Q, Esc       : Exit window.
    [Open3D INFO]     H            : Print help message.
    [Open3D INFO]     P, PrtScn    : Take a screen capture.
    [Open3D INFO]     D            : Take a depth capture.
    [Open3D INFO]     O            : Take a capture of current rendering settings.
    [Open3D INFO]     Alt + Enter  : Toggle between full screen and windowed mode.
    [Open3D INFO] 
    [Open3D INFO]   -- Render mode control --
    [Open3D INFO]     L            : Turn on/off lighting.
    [Open3D INFO]     +/-          : Increase/decrease point size.
    [Open3D INFO]     Ctrl + +/-   : Increase/decrease width of geometry::LineSet.
    [Open3D INFO]     N            : Turn on/off point cloud normal rendering.
    [Open3D INFO]     S            : Toggle between mesh flat shading and smooth shading.
    [Open3D INFO]     W            : Turn on/off mesh wireframe.
    [Open3D INFO]     B            : Turn on/off back face rendering.
    [Open3D INFO]     I            : Turn on/off image zoom in interpolation.
    [Open3D INFO]     T            : Toggle among image render:
    [Open3D INFO]                    no stretch / keep ratio / freely stretch.
    [Open3D INFO] 
    [Open3D INFO]   -- Color control --
    [Open3D INFO]     0..4,9       : Set point cloud color option.
    [Open3D INFO]                    0 - Default behavior, render point color.
    [Open3D INFO]                    1 - Render point color.
    [Open3D INFO]                    2 - x coordinate as color.
    [Open3D INFO]                    3 - y coordinate as color.
    [Open3D INFO]                    4 - z coordinate as color.
    [Open3D INFO]                    9 - normal as color.
    [Open3D INFO]     Ctrl + 0..4,9: Set mesh color option.
    [Open3D INFO]                    0 - Default behavior, render uniform gray color.
    [Open3D INFO]                    1 - Render point color.
    [Open3D INFO]                    2 - x coordinate as color.
    [Open3D INFO]                    3 - y coordinate as color.
    [Open3D INFO]                    4 - z coordinate as color.
    [Open3D INFO]                    9 - normal as color.
    [Open3D INFO]     Shift + 0..4 : Color map options.
    [Open3D INFO]                    0 - Gray scale color.
    [Open3D INFO]                    1 - JET color map.
    [Open3D INFO]                    2 - SUMMER color map.
    [Open3D INFO]                    3 - WINTER color map.
    [Open3D INFO]                    4 - HOT color map.
    [Open3D INFO] 
    [Open3D INFO]   -- Mouse view control --
    [Open3D INFO]     Left button + drag         : Rotate.
    [Open3D INFO]     Ctrl + left button + drag  : Translate.
    [Open3D INFO]     Wheel button + drag        : Translate.
    [Open3D INFO]     Shift + left button + drag : Roll.
    [Open3D INFO]     Wheel                      : Zoom in/out.
    [Open3D INFO] 
    [Open3D INFO]   -- Keyboard view control --
    [Open3D INFO]     [/]          : Increase/decrease field of view.
    [Open3D INFO]     R            : Reset view point.
    [Open3D INFO]     Ctrl/Cmd + C : Copy current view status into the clipboard.
    [Open3D INFO]     Ctrl/Cmd + V : Paste view status from clipboard.
    [Open3D INFO] 
    [Open3D INFO]   -- General control --
    [Open3D INFO]     Q, Esc       : Exit window.
    [Open3D INFO]     H            : Print help message.
    [Open3D INFO]     P, PrtScn    : Take a screen capture.
    [Open3D INFO]     D            : Take a depth capture.
    [Open3D INFO]     O            : Take a capture of current rendering settings.
    [Open3D INFO]     Alt + Enter  : Toggle between full screen and windowed mode.
    [Open3D INFO] 
    [Open3D INFO]   -- Render mode control --
    [Open3D INFO]     L            : Turn on/off lighting.
    [Open3D INFO]     +/-          : Increase/decrease point size.
    [Open3D INFO]     Ctrl + +/-   : Increase/decrease width of geometry::LineSet.
    [Open3D INFO]     N            : Turn on/off point cloud normal rendering.
    [Open3D INFO]     S            : Toggle between mesh flat shading and smooth shading.
    [Open3D INFO]     W            : Turn on/off mesh wireframe.
    [Open3D INFO]     B            : Turn on/off back face rendering.
    [Open3D INFO]     I            : Turn on/off image zoom in interpolation.
    [Open3D INFO]     T            : Toggle among image render:
    [Open3D INFO]                    no stretch / keep ratio / freely stretch.
    [Open3D INFO] 
    [Open3D INFO]   -- Color control --
    [Open3D INFO]     0..4,9       : Set point cloud color option.
    [Open3D INFO]                    0 - Default behavior, render point color.
    [Open3D INFO]                    1 - Render point color.
    [Open3D INFO]                    2 - x coordinate as color.
    [Open3D INFO]                    3 - y coordinate as color.
    [Open3D INFO]                    4 - z coordinate as color.
    [Open3D INFO]                    9 - normal as color.
    [Open3D INFO]     Ctrl + 0..4,9: Set mesh color option.
    [Open3D INFO]                    0 - Default behavior, render uniform gray color.
    [Open3D INFO]                    1 - Render point color.
    [Open3D INFO]                    2 - x coordinate as color.
    [Open3D INFO]                    3 - y coordinate as color.
    [Open3D INFO]                    4 - z coordinate as color.
    [Open3D INFO]                    9 - normal as color.
    [Open3D INFO]     Shift + 0..4 : Color map options.
    [Open3D INFO]                    0 - Gray scale color.
    [Open3D INFO]                    1 - JET color map.
    [Open3D INFO]                    2 - SUMMER color map.
    [Open3D INFO]                    3 - WINTER color map.
    [Open3D INFO]                    4 - HOT color map.
    [Open3D INFO] 
    
    ```

  - Voxel downsampling

    tips:

    Voxel = Volume Pixel

    体素 = 体积像素

    对点云进行预处理—降采样，共有两个步骤

    1. Points are bucketed into voxels
    2. Each occupied voxel generate exactly one  point by averaging all points inside

    ```python
    print("Downsample the point cloud with a voxel of 0.05")
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw_geometries([downpcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
    ```

  - Vertex normal estimation

    顶点法向估计

    点云的法向估计是一个基本的操作，按下N可以看到点云的法向，按下-或+可以控制法向的长度

    **estimate_normals** 为每个点都计算法向

    **KDTreeSearchParamHybrid** KD树搜索

  - Access estimated vertex normal

    ```python
    print("Print a normal vector of the 0th point")
    print(downpcd.normals[0])
    ```

  - Crop point cloud

    ```python
    print("Load a polygon volume and use it to crop the original point cloud")
    vol = o3d.visualization.read_selection_polygon_volume(
        "../../test_data/Crop/cropped.json")
    chair = vol.crop_point_cloud(pcd)
    o3d.visualization.draw_geometries([chair],
                                      zoom=0.7,
                                      front=[0.5439, -0.2333, -0.8060],
                                      lookat=[2.4615, 2.1331, 1.338],
                                      up=[-0.1781, -0.9708, 0.1608])
    ```

  - Paint point cloud

    ```python
    print("Paint chair")
    chair.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([chair],
                                      zoom=0.7,
                                      front=[0.5439, -0.2333, -0.8060],
                                      lookat=[2.4615, 2.1331, 1.338],
                                      up=[-0.1781, -0.9708, 0.1608])
    ```

    **paint_uniform_color** 画出相同的颜色，范围0-1 

  - Point cloud distance

    ```python
    pcd = o3d.io.read_point_cloud("../../test_data/fragment.ply")
    vol = o3d.visualization.read_selection_polygon_volume(
        "../../test_data/Crop/cropped.json")
    chair = vol.crop_point_cloud(pcd)
    
    dists = pcd.compute_point_cloud_distance(chair)
    dists = np.asarray(dists)
    ind = np.where(dists > 0.01)[0]
    pcd_without_chair = pcd.select_by_index(ind)
    o3d.visualization.draw_geometries([pcd_without_chair],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
    ```

  - Bounding volume

    ```python
    aabb = chair.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    obb = chair.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    o3d.visualization.draw_geometries([chair, aabb, obb],
                                      zoom=0.7,
                                      front=[0.5439, -0.2333, -0.8060],
                                      lookat=[2.4615, 2.1331, 1.338],
                                      up=[-0.1781, -0.9708, 0.1608])
    ```

    可以计算点云的边界

  - Convex hull

    凸包

    点云凸包是最小的凸集包含所有点

  - DBSCAN clustering

    

  - Plane segmentation

    

  - Hidden point removal

    

















### Processing

### Interface



reference：

http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html

https://zh.wikipedia.org/wiki/%E9%AB%94%E7%B4%A0