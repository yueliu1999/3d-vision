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

    ```python
  pcl = o3dtut.get_bunny_mesh().sample_points_poisson_disk(number_of_points=2000)
    hull, _ = pcl.compute_convex_hull()
  hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
  o3d.visualization.draw_geometries([pcl, hull_ls])
    ```

    **compute_convex_hull** that computes the convex hull of a point cloud. The implementation is based on Qhull

    
  
  - DBSCAN clustering
  
    ```python
    pcd = o3d.io.read_point_cloud("../../test_data/fragment.ply")
    
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
    
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.455,
                                      front=[-0.4999, -0.1659, -0.8499],
                                      lookat=[2.1813, 2.0619, 2.0999],
                                      up=[0.1204, -0.9852, 0.1215])
    ```
  
    cluster_dbscan are required two parameters:
  
    - eps define the distance to neighbors
    - min_points defines the minimum number of points
  
    return labels
  
  - Plane segmentation
  
    segmentation of geometric primitives from point cloud using RANSAC
  
    ```python
    pcd = o3d.io.read_point_cloud("../../test_data/fragment.pcd")
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.8,
                                      front=[-0.4999, -0.1659, -0.8499],
                                      lookat=[2.1813, 2.0619, 2.0999],
                                      up=[0.1204, -0.9852, 0.1215])
    ```
  
    **segment_plane** 
  
    arguments:
  
    - distance_threshold
  
      the max distance a point can have to an estimated plane to be considered an inlier
  
    - ransac_n
  
      define the num of points that are randomly sampled to estimate a plane
  
    - num_iterations
  
      how often a random plane is sampled and verified
  
    return a,b,c,d
  
     ax+by+cz+d=0
  
    
  
  - Hidden point removal
  
    可以隐藏那些看不到的点云
    
    ```python
    print("Convert mesh to a point cloud and estimate dimensions")
    pcd = o3dtut.get_armadillo_mesh().sample_points_poisson_disk(5000)
    diameter = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    o3d.visualization.draw_geometries([pcd])
    
    print("Define parameters used for hidden_point_removal")
    camera = [0, 0, diameter]
    radius = diameter * 100
    
    print("Get all points that are visible from given view point")
    _, pt_map = pcd.hidden_point_removal(camera, radius)
    
    print("Visualize result")
    pcd = pcd.select_by_index(pt_map)
    o3d.visualization.draw_geometries([pcd])
    ```





- Mesh

  Open3d has data structure for 3d triangle meshs called **TriangleMesh**

  - Read from a ply file

    print its vertices and triangles

    ```python
    print("Testing mesh in Open3D...")
    mesh = o3dtut.get_knot_mesh()
    print(mesh)
    print('Vertices:')
    print(np.asarray(mesh.vertices))
    print('Triangles:')
    print(np.asarray(mesh.triangles))
    ```

    TriangleMesh是三角网格

    有以下属性

    - vertices：顶点
    - triangles：三角形

    

  - Visualize a 3D mesh

    ```python
    print("Try to render a mesh with normals (exist: " +
          str(mesh.has_vertex_normals()) + ") and colors (exist: " +
          str(mesh.has_vertex_colors()) + ")")
    o3d.visualization.draw_geometries([mesh])
    print("A mesh with no normals and no colors does not look good.")
    ```

    这个时候顶点没有法向normal和面face
    
    
    
  - Surface normal estimate

    ```python
    print("Computing normal and rendering it.")
    mesh.compute_vertex_normals()
    print(np.asarray(mesh.triangle_normals))
    o3d.visualization.draw_geometries([mesh])
    ```

    用compute_vertex_normals来计算法向

    

  - Crop mesh

    切掉一半的mesh，可以使用numpy来解决

    ```python
    print("We make a partial mesh of only the first half triangles.")
    mesh1 = copy.deepcopy(mesh)
    mesh1.triangles = o3d.utility.Vector3iVector(
        np.asarray(mesh1.triangles)[:len(mesh1.triangles) // 2, :])
    mesh1.triangle_normals = o3d.utility.Vector3dVector(
        np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals) // 2, :])
    print(mesh1.triangles)
    o3d.visualization.draw_geometries([mesh1])
    ```

    

  - Paint mesh

    可以用paint_uniform_color来显示出颜色

    ```python
    print("Painting the mesh")
    mesh1.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([mesh1])
    ```

    

  - Mesh properties

    
    
  - Mesh filtering

    - Average filter
    
      一个顶点是由其相邻点的均值来计算的
      $$
      v_i = \frac{v_i+\sum_{n\in N^{v_{n}}}}{|N|+1}
      $$
      可以用来降噪
    
      **number_of_iterations** in the function **filter_smooth_simple** define the how often the filter is applied to the mesh
    
      ```python
      print('create noisy mesh')
      mesh_in = o3dtut.get_knot_mesh()
      vertices = np.asarray(mesh_in.vertices)
      noise = 5
      vertices += np.random.uniform(0, noise, size=vertices.shape)
      mesh_in.vertices = o3d.utility.Vector3dVector(vertices)
      mesh_in.compute_vertex_normals()
      o3d.visualization.draw_geometries([mesh_in])
      
      print('filter with average with 1 iteration')
      mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=1)
      mesh_out.compute_vertex_normals()
      o3d.visualization.draw_geometries([mesh_out])
      
      print('filter with average with 5 iterations')
      mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=5)
      mesh_out.compute_vertex_normals()
      o3d.visualization.draw_geometries([mesh_out])
      
      ```
    
      
    
    - Laplacian
      $$
      v_i = v_i \lambda \sum_{n\in N}w_nv_n - v_i
      \\ 
      \lambda \ is \ the \ strength \ of \ the \ filter 
      \\
      w_n \ is \ normalized \ weights \ that \ relate \ to \ the \ distance \ of \ neighboring \ vertices
      $$
      **filter_smooth_laplacian**
    
      - **number_of_iterations**
      - **lambda**
    
      ```python
      print('filter with Laplacian with 10 iterations')
      mesh_out = mesh_in.filter_smooth_laplacian(number_of_iterations=10)
      mesh_out.compute_vertex_normals()
      o3d.visualization.draw_geometries([mesh_out])
      
      print('filter with Laplacian with 50 iterations')
      mesh_out = mesh_in.filter_smooth_laplacian(number_of_iterations=50)
      mesh_out.compute_vertex_normals()
      o3d.visualization.draw_geometries([mesh_out])
      ```
    
      
    
    - Taubin filter
    
      average and laplacian filter is that they lead to a shrinkage of the triangle mesh
    
      tabubin filter showed that the application of two laplacian filters with different labmda parameters can prevent the mesh shrinkage.
    
      **filter_smooth_taubin**
    
      ```python
      print('filter with Taubin with 10 iterations')
      mesh_out = mesh_in.filter_smooth_taubin(number_of_iterations=10)
      mesh_out.compute_vertex_normals()
      o3d.visualization.draw_geometries([mesh_out])
      
      print('filter with Taubin with 100 iterations')
      mesh_out = mesh_in.filter_smooth_taubin(number_of_iterations=100)
      mesh_out.compute_vertex_normals()
      o3d.visualization.draw_geometries([mesh_out])
      ```
    
      
    
  - Sampling

    **sample_points_uniformly** that uniformly samples points from the 3D surface based on the triangle area

    **number_of_points** defines how many points are sampled  from the triangle surface

    ```python
    mesh = o3d.geometry.TriangleMesh.create_sphere()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
    pcd = mesh.sample_points_uniformly(number_of_points=500)
    o3d.visualization.draw_geometries([pcd])
    ```

    

  - Mesh subdivision

    我们可以将三角网格进行细分化，例如直接找到三角形边的中点

    **subdivide_midpoint**

    ```python
    mesh = o3d.geometry.TriangleMesh.create_box()
    mesh.compute_vertex_normals()
    print(
        f'The mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
    )
    o3d.visualization.draw_geometries([mesh], zoom=0.8, mesh_show_wireframe=True)
    mesh = mesh.subdivide_midpoint(number_of_iterations=1)
    print(
        f'After subdivision it has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
    )
    o3d.visualization.draw_geometries([mesh], zoom=0.8, mesh_show_wireframe=True)
    ```

    我们也可以用另外一个subdivision的方法，基于quartic box spline

    ```python
    mesh = o3d.geometry.TriangleMesh.create_sphere()
    mesh.compute_vertex_normals()
    print(
        f'The mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
    )
    o3d.visualization.draw_geometries([mesh], zoom=0.8, mesh_show_wireframe=True)
    mesh = mesh.subdivide_loop(number_of_iterations=2)
    print(
        f'After subdivision it has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
    )
    o3d.visualization.draw_geometries([mesh], zoom=0.8, mesh_show_wireframe=True)
    ```

    

  - Mesh simplification

    可以简化三角网格和顶点

    - Vertex clustering

      pool all vertices that fall into a voxel of a given size to a single vertex

      **simplify_vertex_clustering**

      - voxel_size:

        the size of the voxel grid

      - contraction

        how the vertices are pooled

        **o3d.geometry.SimplificationContraction.Average**

      ```python
      mesh_in = o3dtut.get_bunny_mesh()
      print(
          f'Input mesh has {len(mesh_in.vertices)} vertices and {len(mesh_in.triangles)} triangles'
      )
      o3d.visualization.draw_geometries([mesh_in])
      
      voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 32
      print(f'voxel_size = {voxel_size:e}')
      mesh_smp = mesh_in.simplify_vertex_clustering(
          voxel_size=voxel_size,
          contraction=o3d.geometry.SimplificationContraction.Average)
      print(
          f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
      )
      o3d.visualization.draw_geometries([mesh_smp])
      
      voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 16
      print(f'voxel_size = {voxel_size:e}')
      mesh_smp = mesh_in.simplify_vertex_clustering(
          voxel_size=voxel_size,
          contraction=o3d.geometry.SimplificationContraction.Average)
      print(
          f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
      )
      o3d.visualization.draw_geometries([mesh_smp])
      ```

      

      

    - Mesh decimation

      最小化error quadrics (distances to neighboring planes)

      **target_number_of_triangles** defines the stopping critera of the decimation algorithm

      ```python
      mesh_smp = mesh_in.simplify_quadric_decimation(target_number_of_triangles=6500)
      print(
          f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
      )
      o3d.visualization.draw_geometries([mesh_smp])
      
      mesh_smp = mesh_in.simplify_quadric_decimation(target_number_of_triangles=1700)
      print(
          f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles'
      )
      o3d.visualization.draw_geometries([mesh_smp])
      ```

      

    - Connected components

      可以对不同mesh进行拼接，可以去噪声

      **cluster_connected_triangles**

      - triangle_clusters
      - cluster_n_triangles
      - cluster_area

      ```python
      print("Generate data")
      mesh = o3dtut.get_bunny_mesh().subdivide_midpoint(number_of_iterations=2)
      vert = np.asarray(mesh.vertices)
      min_vert, max_vert = vert.min(axis=0), vert.max(axis=0)
      for _ in range(30):
          cube = o3d.geometry.TriangleMesh.create_box()
          cube.scale(0.005, center=cube.get_center())
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
      ```

      ```python
      print("Cluster connected triangles")
      with o3d.utility.VerbosityContextManager(
              o3d.utility.VerbosityLevel.Debug) as cm:
          triangle_clusters, cluster_n_triangles, cluster_area = (
              mesh.cluster_connected_triangles())
      triangle_clusters = np.asarray(triangle_clusters)
      cluster_n_triangles = np.asarray(cluster_n_triangles)
      cluster_area = np.asarray(cluster_area)
      ```





- RGBD images

  RGBImage由两个images组成

  - RGBDImage.depth
  - RGBDImage.color

  可以读入深度图和彩色图

  ```python
  print("Read Redwood dataset")
  color_raw = o3d.io.read_image("../../test_data/RGBD/color/00000.jpg")
  depth_raw = o3d.io.read_image("../../test_data/RGBD/depth/00000.png")
  rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
      color_raw, depth_raw)
  print(rgbd_image)
  ```

  可以用**create_rgbd_image_from_color_and_depth**函数来合成RGBDImage

  ```python
  plt.subplot(1, 2, 1)
  plt.title('Redwood grayscale image')
  plt.imshow(rgbd_image.color)
  plt.subplot(1, 2, 2)
  plt.title('Redwood depth image')
  plt.imshow(rgbd_image.depth)
  plt.show()
  ```

  RGBDImage可以转换为点云，需要给出相机参数

  ```python
  pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
      rgbd_image,
      o3d.camera.PinholeCameraIntrinsic(
          o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
  # Flip it, otherwise the pointcloud will be upside down
  pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
  o3d.visualization.draw_geometries([pcd], zoom=0.5)
  ```

  **create_from_rgbd_image**

  相机参数是**PinholeCameraIntrinsicParameters.PrimeSenseDefault**

  - resolution: 640pixel*480pixel

  - focal length: (fx, fy) = (525.0, 525.0)

  - optical center: (cx, cy) = (319.5 239.5)

  - extrinsic parameter: identity matrix

    

- KDTree

  **build KDTree from point cloud**

  preprocessing step for the following nearest neighbor queries

  ```python
  print("Testing kdtree in Open3D...")
  print("Load a point cloud and paint it gray.")
  pcd = o3d.io.read_point_cloud("../../test_data/Feature/cloud_bin_0.pcd")
  pcd.paint_uniform_color([0.5, 0.5, 0.5])
  pcd_tree = o3d.geometry.KDTreeFlann(pcd)
  ```

  **Find neighboring points**
  
  ```python
  print("Paint the 1500th point red.")
  pcd.colors[1500] = [1, 0, 0]
  ```
  
  **Using search_knn_vector_3d**
  
  ```python
  print("Find its 200 nearest neighbors, and paint them blue.")
  [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], 200)
  np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
  ```
  
  **Using search_radius_vector_3d**
  
  ```python
  print("Find its neighbors with distance less than 0.2, and paint them green.")
  [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], 0.2)
  np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]
  ```



### Processing

- File IO

  - Point cloud
  - Mesh
  - Image

  

- Point cloud outlier removal

  从扫描设备中获取的点云，会有一些噪声，我们需要将这些噪声去除掉

  - Prepare input data

    降采样使用**voxel_downsample**

    ```python
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_2.pcd")
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
    
    print("Downsample the point cloud with a voxel of 0.02")
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    o3d.visualization.draw_geometries([voxel_down_pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
    ```

    降采样使用**uniform down sample**

    ```python
    print("Every 5th points are selected")
    uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
    o3d.visualization.draw_geometries([uni_down_pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
    ```

    

  - Select down sample

    ```python
    def display_inlier_outlier(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)
    
        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                          zoom=0.3412,
                                          front=[0.4257, -0.2125, -0.8795],
                                          lookat=[2.6172, 2.0475, 1.532],
                                          up=[-0.0694, -0.9768, 0.2024])
    ```

    

  - Statistical outlier removal

    移除那些距离点云平均值邻居很远的点

    有两个参数

    - nb_neighbor

      计算n个邻居的平均距离

    - std_ratio

      标准差的阈值

    ```python
    print("Statistical oulier removal")
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=2.0)
    display_inlier_outlier(voxel_down_pcd, ind)
    ```

  - Radius outlier removal

    给出一个球，如果临近点很少则认为是outlier

    有两个参数

    - nb_points

      选出最小的数量在球的周围

    - radius

      radius of sphere

    ```python
    print("Radius oulier removal")
    cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    display_inlier_outlier(voxel_down_pcd, ind)
    ```



- Voxelization

  点云和三角网格是非常灵活但是不规则的几何类型，体素网格是另外一种3d形式

  Open3D有**VoxelGrid**几何类型可以被使用于voxel grid

  

  - From triangle mesh

    ```python
    print('input')
    mesh = o3dtut.get_bunny_mesh()
    # fit to unit cube
    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
               center=mesh.get_center())
    o3d.visualization.draw_geometries([mesh])
    
    print('voxelization')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                                  voxel_size=0.05)
    o3d.visualization.draw_geometries([voxel_grid])
    ```

  - From point cloud

    ```python
    print('input')
    N = 2000
    pcd = o3dtut.get_armadillo_mesh().sample_points_poisson_disk(N)
    # fit to unit cube
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
              center=pcd.get_center())
    pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
    o3d.visualization.draw_geometries([pcd])
    
    print('voxelization')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                voxel_size=0.05)
    o3d.visualization.draw_geometries([voxel_grid])
    ```

  - Inclusion test

    ```python
    queries = np.asarray(pcd.points)
    output = voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))
    print(output[:10])
    ```

  - Voxel carving

  

- Surface reconstruction

  a dense 3D geometry, i.e., a triangle mesh

  face reconstruction method

  - Alpha shapes

    **create_from_point_cloud_alpha_shape**

    parameter **alpha**

    ```python
    mesh = o3dtut.get_bunny_mesh()
    pcd = mesh.sample_points_poisson_disk(750)
    o3d.visualization.draw_geometries([pcd])
    alpha = 0.03
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    ```

    ```python
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    for alpha in np.logspace(np.log10(0.5), np.log10(0.01), num=4):
        print(f"alpha={alpha:.3f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha, tetra_mesh, pt_map)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    ```

    

  - Ball pivoting

    **create_from_point_cloud_ball_pivoting**

    parameter radii

    ```python
    gt_mesh = o3dtut.get_bunny_mesh()
    gt_mesh.compute_vertex_normals()
    pcd = gt_mesh.sample_points_poisson_disk(3000)
    o3d.visualization.draw_geometries([pcd])
    ```

    ```python
    radii = [0.005, 0.01, 0.02, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    o3d.visualization.draw_geometries([pcd, rec_mesh])
    ```

  - Poisson surface reconstruction

    ```python
    pcd = o3dtut.get_eagle_pcd()
    print(pcd)
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.664,
                                      front=[-0.4761, -0.4698, -0.7434],
                                      lookat=[1.8900, 3.2596, 0.9284],
                                      up=[0.2304, -0.8825, 0.4101])
    print('run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)
    print(mesh)
    o3d.visualization.draw_geometries([mesh],
                                      zoom=0.664,
                                      front=[-0.4761, -0.4698, -0.7434],
                                      lookat=[1.8900, 3.2596, 0.9284],
                                      up=[0.2304, -0.8825, 0.4101])
    ```

    

  - Normal estimation

    ```python
    gt_mesh = o3dtut.get_bunny_mesh()
    pcd = gt_mesh.sample_points_poisson_disk(5000)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros(
        (1, 3)))  # invalidate existing normals
    
    pcd.estimate_normals()
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    pcd.orient_normals_consistent_tangent_plane(100)
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    ```

  

- Transformation

  - translate
    $$
    v_t = v+t
    $$

    ```python
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh_tx = copy.deepcopy(mesh).translate((1.3, 0, 0))
    mesh_ty = copy.deepcopy(mesh).translate((0, 1.3, 0))
    print(f'Center of mesh: {mesh.get_center()}')
    print(f'Center of mesh tx: {mesh_tx.get_center()}')
    print(f'Center of mesh ty: {mesh_ty.get_center()}')
    o3d.visualization.draw_geometries([mesh, mesh_tx, mesh_ty])
    ```

    - get_center

      返回mean of triangle mesh vertices

    - relative

      如果是相对位置，则进行tranlate，如果不是相对，则直接转换到所指定的位置

      

      

  - rotate

    a rotation matrix **R**

    - convert from Euler angles

      **get_rotation_matrix_from_xyz**

    - convert from axis-angle representation

      **get_rotation_matrix_from_axis_angle**

    - convert from quaternions

      **get_rotation_matirx_from_quaternion**

    ```python
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh_r = copy.deepcopy(mesh)
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4))
    mesh_r.rotate(R, center=(0, 0, 0))
    o3d.visualization.draw_geometries([mesh, mesh_r])
    ```

    

  - scale
    $$
    v_s = s  \cdot v
    $$
    

    ```python
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh_s = copy.deepcopy(mesh).translate((2, 0, 0))
    mesh_s.scale(0.5, center=mesh_s.get_center())
    o3d.visualization.draw_geometries([mesh, mesh_s])
    ```

    ```python
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh_s = copy.deepcopy(mesh).translate((2, 1, 0))
    mesh_s.scale(0.5, center=(0, 0, 0))
    o3d.visualization.draw_geometries([mesh, mesh_s])
    ```

    

  - transform

    4*4 homogeneous transformation matrix

    ```python
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    T = np.eye(4)
    T[:3, :3] = mesh.get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
    T[0, 3] = 1
    T[1, 3] = 1.3
    print(T)
    mesh_t = copy.deepcopy(mesh).transform(T)
    o3d.visualization.draw_geometries([mesh, mesh_t])
    ```





- Mesh deformation

  as-rigid-as-possible method by that optimizes the following energy function:
  $$
  \sum_i\sum_{j\in N(i)}w_{ij}||(p_i^{'}-p_j^{'})-R_i(p_i-p_j)||^2
  $$

  ```python
  mesh = o3dtut.get_armadillo_mesh()
  
  vertices = np.asarray(mesh.vertices)
  static_ids = [idx for idx in np.where(vertices[:, 1] < -30)[0]]
  static_pos = []
  for id in static_ids:
      static_pos.append(vertices[id])
  handle_ids = [2490]
  handle_pos = [vertices[2490] + np.array((-40, -40, -40))]
  constraint_ids = o3d.utility.IntVector(static_ids + handle_ids)
  constraint_pos = o3d.utility.Vector3dVector(static_pos + handle_pos)
  
  with o3d.utility.VerbosityContextManager(
          o3d.utility.VerbosityLevel.Debug) as cm:
      mesh_prime = mesh.deform_as_rigid_as_possible(constraint_ids,
                                                    constraint_pos,
                                                    max_iter=50)
  ```



- Intrinsic shape signature(ISS)

  ```python
  # Compute ISS Keypoints on Armadillo
  mesh = o3dtut.get_armadillo_mesh()
  pcd = o3d.geometry.PointCloud()
  pcd.points = mesh.vertices
  
  tic = time.time()
  keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
  toc = 1000 * (time.time() - tic)
  print("ISS Computation took {:.0f} [ms]".format(toc))
  
  mesh.compute_vertex_normals()
  mesh.paint_uniform_color([0.5, 0.5, 0.5])
  keypoints.paint_uniform_color([1.0, 0.75, 0.0])
  o3d.visualization.draw_geometries([keypoints, mesh], front=[0, 0, -1.0])
  ```



### Interface

- Working with numpy

  Open3d中所有的数据都和Numpy兼容

  生成一些numpy的数据

  ```python
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
  ```

  - from numpy to open3d.PointCloud

    ```python
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    ```

    **open3d.PointCloud.colors**和**open3d.PointCloud.normals**都可以被赋值

  - from open3d.PointCloud to Numpy

    ```python
    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("../../test_data/sync.ply")
    
    # Convert Open3D.o3d.geometry.PointCloud to numpy array
    xyz_load = np.asarray(pcd_load.points)
    print('xyz_load')
    print(xyz_load)
    o3d.visualization.draw_geometries([pcd_load])
    ```

    

### 



reference：

http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html

https://zh.wikipedia.org/wiki/%E9%AB%94%E7%B4%A0