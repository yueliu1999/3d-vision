## Pipelines

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
- Input
- Point-to-plane ICP
- Colored point cloud registration
- 





### Global registration



### Multiway registraion



### RGBD integration







