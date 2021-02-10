Pipelines

### ICP registration

- Helper visualization function

  ```python
  def draw_registration_result(source, target, transformation):
      source_temp = copy.deepcopy(source)
      target_temp = copy.deepcopy(target)
      source_temp.paint_uniform_color([1, 0.706, 0])
      target_temp.paint_uniform_color([0, 0.651, 0.929])
      source_temp.transform(transformation)
      o3d.visualization.draw_geometries([source_temp, target_temp],
                                        zoom=0.4459,
                                        front=[0.9288, -0.2951, -0.2242],
                                        lookat=[1.6784, 2.0612, 1.4451],
                                        up=[-0.3402, -0.9189, -0.1996])
  ```

- Input

  **evaluate_registration** calculates two main metrics

  - fitness

    用于度量overlap的面积

    inliers的数量/target 数量

    ```python
    source = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_0.pcd")
    target = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_1.pcd")
    threshold = 0.02
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    draw_registration_result(source, target, trans_init)
    ```

  - inlier_rmse

    度量RMSE of all correspondences

    ```python
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(evaluation)
    ```



- Point-to-point ICP

  ICP有两个步骤

  1. 找到correspondence set

     K = {(p,q)} from target points P and source point cloud Q transformed with current transformation matrix T

  2. 更新T，通过最小化目标函数E(T)，defined over the correspondence set K

  不同的ICP使用不同的目标函数

  

  对于 point-to-point ICP algorithm
  $$
  E(T) = \sum_{(p,q) \in k}||p-Tq||^2
  $$
  **TransformationEstimationPointToPoint** 类可以提供点点ICP计算残差和雅可比矩阵

  **registration_icp** 函数可以进行ICP 

  ```python
  print("Apply point-to-point ICP")
  reg_p2p = o3d.pipelines.registration.registration_icp(
      source, target, threshold, trans_init,
      o3d.pipelines.registration.TransformationEstimationPointToPoint())
  print(reg_p2p)
  print("Transformation is:")
  print(reg_p2p.transformation)
  draw_registration_result(source, target, reg_p2p.transformation)
  ```

  

- point-to-plane ICP

  use a different objective function
  $$
  E(T) = \sum_{(p,q)\in k}((p-Tq)\cdot n_p)^2
  $$
  faster convergence speed than the point-to-point ICP

  **TransformationEstimationPointToPlane** 类可以提供点面ICP计算残差和雅可比矩阵

  **registration_icp** 函数可以进行点-点ICP 

  ```python
  print("Apply point-to-plane ICP")
  reg_p2l = o3d.pipelines.registration.registration_icp(
      source, target, threshold, trans_init,
      o3d.pipelines.registration.TransformationEstimationPointToPlane())
  print(reg_p2l)
  print("Transformation is:")
  print(reg_p2l.transformation)
  draw_registration_result(source, target, reg_p2l.transformation)
  ```



### Robust kernels

context outlier rejection

implemented for the PointToPlane ICP

- Input data

  ```python
  def draw_registration_result(source, target, transformation):
      source_temp = copy.deepcopy(source)
      target_temp = copy.deepcopy(target)
      source_temp.paint_uniform_color([1, 0.706, 0])
      target_temp.paint_uniform_color([0, 0.651, 0.929])
      source_temp.transform(transformation)
      o3d.visualization.draw_geometries([source_temp, target_temp],
                                        zoom=0.4459,
                                        front=[0.9288, -0.2951, -0.2242],
                                        lookat=[1.6784, 2.0612, 1.4451],
                                        up=[-0.3402, -0.9189, -0.1996])
  ```

  ```python
  source = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_0.pcd")
  target = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_1.pcd")
  trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                           [-0.139, 0.967, -0.215, 0.7],
                           [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
  draw_registration_result(source, target, trans_init)
  ```

- Point-to-plane ICP using robust kernels

  基本的点面ICP的目标函数是
  $$
  E(T) = \sum_{(p,q)\in K}((p-Tq)\cdot n_{p})^2
  \\ 
  n_p \ the \ normal \ of \ point \ p \ and \ K \ is \ the \ correspondence \ set
  \\ 
$$
重新写该残差
$$
  E(T) = \sum_{(p,q) \in K}((p-Tq) \cdot n_p)^2 = \sum_{i=1}^N(r_i(T))^2
$$
  可以使用iteratively reweighted least-squares 的方法进行求解
$$
  E(T) = \sum_{i=1}^Nw_i(r_i(T))^2
$$
  
- Outlier Rejection with Robust Kernels

  主要的思想是降低残差大的权重，因为有可能他们是outliers
$$
  E(T) = \sum_{(p,q) \in K} \rho((p-Tq) \cdot n_p) = \sum_{i=1}^N \rho(r_i(T))
  \\
  \rho(r) \ is \ called \ the \ robust \ loss 
$$
  minimize the objective function using Gauss-Newton and determine increments by iteratively solving:
$$
  (J^TWJ)^{-1}J^TW_{\vec{r}}
  \\
  W \in R^{N \times N}
$$
  ```python
  def apply_noise(pcd, mu, sigma):
      noisy_pcd = copy.deepcopy(pcd)
      points = np.asarray(noisy_pcd.points)
      points += np.random.normal(mu, sigma, size=points.shape)
      noisy_pcd.points = o3d.utility.Vector3dVector(points)
      return noisy_pcd
  
  
  mu, sigma = 0, 0.1  # mean and standard deviation
  source_noisy = apply_noise(source, mu, sigma)
  
  print("Source PointCloud + noise:")
  o3d.visualization.draw_geometries([source_noisy],
                                    zoom=0.4459,
                                    front=[0.353, -0.469, -0.809],
                                    lookat=[2.343, 2.217, 1.809],
                                    up=[-0.097, -0.879, 0.467])
  ```

  ```python
  threshold = 0.02
  print("Vanilla point-to-plane ICP, threshold={}:".format(threshold))
  p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane()
  reg_p2l = o3d.pipelines.registration.registration_icp(source_noisy, target,
                                                        threshold, trans_init,
                                                        p2l)
  
  print(reg_p2l)
  print("Transformation is:")
  print(reg_p2l.transformation)
  draw_registration_result(source, target, reg_p2l.transformation)
  ```

  ```python
  print("Robust point-to-plane ICP, threshold={}:".format(threshold))
  loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
  print("Using robust loss:", loss)
  p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
  reg_p2l = o3d.pipelines.registration.registration_icp(source_noisy, target,
                                                        threshold, trans_init,
                                                        p2l)
  print(reg_p2l)
  print("Transformation is:")
  print(reg_p2l.transformation)
  draw_registration_result(source, target, reg_p2l.transformation)
  ```



### Colored point cloud registration

ICP variant that use

- gemetry
- color

算法会更加鲁棒和准确，速度也是可以接受的

- Helper visualization function

  ```python
  def draw_registration_result_original_color(source, target, transformation):
      source_temp = copy.deepcopy(source)
      source_temp.transform(transformation)
      o3d.visualization.draw_geometries([source_temp, target],
                                        zoom=0.5,
                                        front=[-0.2458, -0.8088, 0.5342],
                                        lookat=[1.7745, 2.2305, 0.9787],
                                        up=[0.3109, -0.5878, -0.7468])
  ```

- Input

  ```python
  print("1. Load two point clouds and show initial pose")
  source = o3d.io.read_point_cloud("../../test_data/ColoredICP/frag_115.ply")
  target = o3d.io.read_point_cloud("../../test_data/ColoredICP/frag_116.ply")
  
  # draw initial alignment
  current_transformation = np.identity(4)
  draw_registration_result_original_color(source, target, current_transformation)
  ```

- Point-to-plane ICP

  misaligned green triangle textures

  ```python
  # point to plane ICP
  current_transformation = np.identity(4)
  print("2. Point-to-plane ICP registration is applied on original point")
  print("   clouds to refine the alignment. Distance threshold 0.02.")
  result_icp = o3d.pipelines.registration.registration_icp(
      source, target, 0.02, current_transformation,
      o3d.pipelines.registration.TransformationEstimationPointToPlane())
  print(result_icp)
  draw_registration_result_original_color(source, target,
                                          result_icp.transformation)
  ```

  

- Colored point cloud registration

  code function:

  **registration_colored_icp**

  it runs ICP iterations with a joint optimization objective:
$$
  E(T) = (1-\delta)E_C(T)+\delta E_G(T)
  \\
  E(C) \ is \ the \ photometric \ term
  \\ 
  E(G) \ is \ the \ geometric \ term

$$
  the geometric term $E_G$ is the same as the point-to-plane ICP objective:
$$
  E_G(T) = \sum_{(p,q) \in K}((p-Tq) \cdot n_p)^2
$$
the color term $E_C$ measures the difference between the color of point q and the color of its projection on the tangent plane of p
$$
  E_C(T) = \sum_{(p,q) \in k}(C_p(f(Tq))-C(q))^2
  $$

  ```python
  # colored pointcloud registration
  # This is implementation of following paper
  # J. Park, Q.-Y. Zhou, V. Koltun,
  # Colored Point Cloud Registration Revisited, ICCV 2017
  voxel_radius = [0.04, 0.02, 0.01]
  max_iter = [50, 30, 14]
  current_transformation = np.identity(4)
  print("3. Colored point cloud registration")
  for scale in range(3):
      iter = max_iter[scale]
      radius = voxel_radius[scale]
      print([iter, radius, scale])
  
      print("3-1. Downsample with a voxel size %.2f" % radius)
      source_down = source.voxel_down_sample(radius)
      target_down = target.voxel_down_sample(radius)
  
      print("3-2. Estimate normal.")
      source_down.estimate_normals(
          o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
      target_down.estimate_normals(
          o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
  
      print("3-3. Applying colored point cloud registration")
      result_icp = o3d.pipelines.registration.registration_colored_icp(
          source_down, target_down, radius, current_transformation,
          o3d.pipelines.registration.TransformationEstimationForColoredICP(),
          o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=iter))
      current_transformation = result_icp.transformation
      print(result_icp)
  draw_registration_result_original_color(source, target,
                                          result_icp.transformation)
  ```

  一共有三个步骤：

  1. 下采样

     **voexl_down_sample**

  2. 估计法向

     normal estimation

  3. 进行带颜色信息的点云注册ICP

     **registration_colored_icp**

     lambda_geometric determines the $\lambda \in[0,1]$ in the overall energy $\lambda E_G+(1-\lambda)E_C$





### Global registration

both **ICP registration** and **Colored point cloud registration** are known as **local registration** 

因为他们都依赖于一个初始的粗对准

另外一个点云注册的方法就是global registration，全局注册不需要有初始的alignment

经常作为local registration的初始值

- Visualization

  ```python
  def draw_registration_result(source, target, transformation):
      source_temp = copy.deepcopy(source)
      target_temp = copy.deepcopy(target)
      source_temp.paint_uniform_color([1, 0.706, 0])
      target_temp.paint_uniform_color([0, 0.651, 0.929])
      source_temp.transform(transformation)
      o3d.visualization.draw_geometries([source_temp, target_temp],
                                        zoom=0.4559,
                                        front=[0.6452, -0.3036, -0.7011],
                                        lookat=[1.9892, 2.0208, 1.8945],
                                        up=[-0.2779, -0.9482, 0.1556])
  ```

- Extract geometric feature

  - downsample 
  - estimate normals
  - compute a FPFH feature

  ```python
  def preprocess_point_cloud(pcd, voxel_size):
      print(":: Downsample with a voxel size %.3f." % voxel_size)
      pcd_down = pcd.voxel_down_sample(voxel_size)
  
      radius_normal = voxel_size * 2
      print(":: Estimate normal with search radius %.3f." % radius_normal)
      pcd_down.estimate_normals(
          o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
  
      radius_feature = voxel_size * 5
      print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
      pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
          pcd_down,
          o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
      return pcd_down, pcd_fpfh
  ```

- Input

  misaligned with identification matrix

  ```python
  def prepare_dataset(voxel_size):
      print(":: Load two point clouds and disturb initial pose.")
      source = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_0.pcd")
      target = o3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_1.pcd")
      trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
      source.transform(trans_init)
      draw_registration_result(source, target, np.identity(4))
  
      source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
      target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
      return source, target, source_down, target_down, source_fpfh, target_fpfh
  ```

- RANSAC

  **ransac_n** random points are picked from the source point cloud

  pruning algorithm，剪枝:

  - **CorrespondenceCheckerBasedOnDistance**

    检查点是否离得太近了

  - **CorrespondenceCheckerBasedOnEdgeLength**

  - **CorrespondenceCheckerBasedOnNormal**

    计算法向的弧度

  核心函数：

  **registration_ransac_based_on_feature_matching**

  参数：

  **RANSACConvergenceCriteria**

  ```python
  def execute_global_registration(source_down, target_down, source_fpfh,
                                  target_fpfh, voxel_size):
      distance_threshold = voxel_size * 1.5
      print(":: RANSAC registration on downsampled point clouds.")
      print("   Since the downsampling voxel size is %.3f," % voxel_size)
      print("   we use a liberal distance threshold %.3f." % distance_threshold)
      result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
          source_down, target_down, source_fpfh, target_fpfh, True,
          distance_threshold,
          o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
          3, [
              o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                  0.9),
              o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                  distance_threshold)
          ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
      return result
  ```

  ```python
  result_ransac = execute_global_registration(source_down, target_down,
                                              source_fpfh, target_fpfh,
                                              voxel_size)
  print(result_ransac)
  draw_registration_result(source_down, target_down, result_ransac.transformation)
  ```

- Local refinement

  全局的registration只能是在严重的降采样下进行，而且结果也不是很好，所以需要用local registration进行refine

  包括

  - point-to-point
  - point-to-plane

  



### Fast global registration

RANSAC会花费太多的时间

Q.-Y. Zhou, J. Park, and V. Koltun, Fast Global Registration, ECCV, 2016.

- Input

  ```python
  voxel_size = 0.05  # means 5cm for the dataset
  source, target, source_down, target_down, source_fpfh, target_fpfh = \
          prepare_dataset(voxel_size)
  ```

- Baseline

  ```python
  start = time.time()
  result_ransac = execute_global_registration(source_down, target_down,
                                              source_fpfh, target_fpfh,
                                              voxel_size)
  print("Global registration took %.3f sec.\n" % (time.time() - start))
  print(result_ransac)
  draw_registration_result(source_down, target_down, result_ransac.transformation)
  ```

- Fast global registration

  ```python
  def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                       target_fpfh, voxel_size):
      distance_threshold = voxel_size * 0.5
      print(":: Apply fast global registration with distance threshold %.3f" \
              % distance_threshold)
      result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
          source_down, target_down, source_fpfh, target_fpfh,
          o3d.pipelines.registration.FastGlobalRegistrationOption(
              maximum_correspondence_distance=distance_threshold))
      return result
  ```

  ```python
  start = time.time()
  result_fast = execute_fast_global_registration(source_down, target_down,
                                                 source_fpfh, target_fpfh,
                                                 voxel_size)
  print("Fast global registration took %.3f sec.\n" % (time.time() - start))
  print(result_fast)
  draw_registration_result(source_down, target_down, result_fast.transformation)
  ```

  



### Multiway registraion

多路注册，

输入是一个几何的集合，比如点云或者是RGBD images $\{ P_i\}$

输出是一个rigid transformation$ \{ {T_i}\}$

可以通过计算出转换后的点云可以align到一起

- Input

  从文件中读入三个点云

  点云需要进行下采样和可视化

  

- Pose graph

  位姿图 pose graph，

  有两个元素

  - nodes

    是一个几何，比如点云

  - edges

    是一个transformation，讲source align到target

    是一个point-to-plane ICP

  

  两两注册会发生错误，错误对齐的数量可能会超过正确对齐的数量

  因此可以将位姿图分为两部分

  - Odometry edges

    连接比较接近相邻的节点，通过local registration 比如ICP来进行align

  - Loop closure edges

    连接任意的节点，可以不相邻，通过global registration 进行align

  除了transformation，用户可以设置一个information matrix 为每一个edge

  

  构建pose graph图：

  ```python
  def pairwise_registration(source, target):
      print("Apply point-to-plane ICP")
      icp_coarse = o3d.pipelines.registration.registration_icp(
          source, target, max_correspondence_distance_coarse, np.identity(4),
          o3d.pipelines.registration.TransformationEstimationPointToPlane())
      icp_fine = o3d.pipelines.registration.registration_icp(
          source, target, max_correspondence_distance_fine,
          icp_coarse.transformation,
          o3d.pipelines.registration.TransformationEstimationPointToPlane())
      transformation_icp = icp_fine.transformation
      information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
          source, target, max_correspondence_distance_fine,
          icp_fine.transformation)
      return transformation_icp, information_icp
  
  
  def full_registration(pcds, max_correspondence_distance_coarse,
                        max_correspondence_distance_fine):
      pose_graph = o3d.pipelines.registration.PoseGraph()
      odometry = np.identity(4)
      pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
      n_pcds = len(pcds)
      for source_id in range(n_pcds):
          for target_id in range(source_id + 1, n_pcds):
              transformation_icp, information_icp = pairwise_registration(
                  pcds[source_id], pcds[target_id])
              print("Build o3d.pipelines.registration.PoseGraph")
              if target_id == source_id + 1:  # odometry case
                  odometry = np.dot(transformation_icp, odometry)
                  pose_graph.nodes.append(
                      o3d.pipelines.registration.PoseGraphNode(
                          np.linalg.inv(odometry)))
                  pose_graph.edges.append(
                      o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                               target_id,
                                                               transformation_icp,
                                                               information_icp,
                                                               uncertain=False))
              else:  # loop closure case
                  pose_graph.edges.append(
                      o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                               target_id,
                                                               transformation_icp,
                                                               information_icp,
                                                               uncertain=True))
      return pose_graph
  ```

  ```python
  print("Full registration ...")
  max_correspondence_distance_coarse = voxel_size * 15
  max_correspondence_distance_fine = voxel_size * 1.5
  with o3d.utility.VerbosityContextManager(
          o3d.utility.VerbosityLevel.Debug) as cm:
      pose_graph = full_registration(pcds_down,
                                     max_correspondence_distance_coarse,
                                     max_correspondence_distance_fine)
  ```

  优化pose graph

  BA

  可以选择算法

  - GlobalOptimizationGaussNewton
  - GlobalOptimizationLevenbergMarquardt

  

  ```python
  print("Optimizing PoseGraph ...")
  option = o3d.pipelines.registration.GlobalOptimizationOption(
      max_correspondence_distance=max_correspondence_distance_fine,
      edge_prune_threshold=0.25,
      reference_node=0)
  with o3d.utility.VerbosityContextManager(
          o3d.utility.VerbosityLevel.Debug) as cm:
      o3d.pipelines.registration.global_optimization(
          pose_graph,
          o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
          o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
          option)
  ```

  

  

- Visualize optimization

  ```python
  print("Transform points and display")
  for point_id in range(len(pcds_down)):
      print(pose_graph.nodes[point_id].pose)
      pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
  o3d.visualization.draw_geometries(pcds_down,
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])
  ```

- Make a combined point cloud



### RGBD integration

- Read trajectory from .log file

  RGBD数据生成网络，TSDF算法

  使用了**read_trajectory**从.log文件种读取相机的轨迹，一个.log文件的实例如下

  ```
  # examples/TestData/RGBD/odometry.log
  0   0   1
  1   0   0   2
  0   1   0   2
  0   0   1 -0.3
  0   0   0   1
  1   1   2
  0.999988  3.08668e-005  0.0049181  1.99962
  -8.84184e-005  0.999932  0.0117022  1.97704
  -0.0049174  -0.0117024  0.999919  -0.300486
  0  0  0  1
  ```

  

  读入相机的姿态

  ```python
  class CameraPose:
  
      def __init__(self, meta, mat):
          self.metadata = meta
          self.pose = mat
  
      def __str__(self):
          return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
              "Pose : " + "\n" + np.array_str(self.pose)
  
  
  def read_trajectory(filename):
      traj = []
      with open(filename, 'r') as f:
          metastr = f.readline()
          while metastr:
              metadata = list(map(int, metastr.split()))
              mat = np.zeros(shape=(4, 4))
              for i in range(4):
                  matstr = f.readline()
                  mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
              traj.append(CameraPose(metadata, mat))
              metastr = f.readline()
      return traj
  ```

- TSDF volume integration

  - UniformTSDFVolume

  - ScalableTSDFVolume，使用了层次结构，支持更大的场景，具有许多参数

    - voxel_length = 4.0/512.0，means a single voxel size for TSDF volume is $\frac{4.0m}{512.0} = 7.8125mm$, 这个值越小，TSDF volume的分辨率越高，但是合成的结果会受到深度噪声的影响

    - sdf_trunc = 0.04

       the truncation value for the signed distance function (SDF)

    - color_type = TSDFVolumeColorType.RGB8

  ```python
  volume = o3d.pipelines.integration.ScalableTSDFVolume(
      voxel_length=4.0 / 512.0,
      sdf_trunc=0.04,
      color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
  
  for i in range(len(camera_poses)):
      print("Integrate {:d}-th image into the volume.".format(i))
      color = o3d.io.read_image("../../test_data/RGBD/color/{:05d}.jpg".format(i))
      depth = o3d.io.read_image("../../test_data/RGBD/depth/{:05d}.png".format(i))
      rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
          color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
      volume.integrate(
          rgbd,
          o3d.camera.PinholeCameraIntrinsic(
              o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
          np.linalg.inv(camera_poses[i].pose))
  ```

- Extract a mesh

  使用了marching cubes algorithm

  ```python
  print("Extract a triangle mesh from the volume and visualize it.")
  mesh = volume.extract_triangle_mesh()
  mesh.compute_vertex_normals()
  o3d.visualization.draw_geometries([mesh],
                                    front=[0.5297, -0.1873, -0.8272],
                                    lookat=[2.0712, 2.0312, 1.7251],
                                    up=[-0.0558, -0.9809, 0.1864],
                                    zoom=0.47)
  ```

  

### RGBD Odometry



### Color Map Optimization













