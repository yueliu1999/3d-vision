## Minth OpenMVG Pipeline

## 相机标定

#### 标定相机初始的内外参

使用Calibra进行相机标定，张正友标定法：

标定

- 内参：fx，fy，cx，cy
- 外参：R，t
- 畸变系数：
  - 径向畸变：k1，k2，k3
  - 切向畸变：p1，p2



reference：

https://zhuanlan.zhihu.com/p/94244568



#### 模板生成





#### 2d检测

1. 图像去畸变，利用初始的畸变系数：k1，k2，k3，p1，p2

2. 利用halcon进行模板匹配：

   1. ROI：获取到每个ROI（手工标注的ROI）

      格式：camera-CCDn.json代表CCDn相机view下的ROI

      - name：测点名字
      - lx：左上角点的x
      - ly：左上角点的y
      - rx：右下角点的x
      - ry：右下角点的y

   2. 图像预处理：

      - 将ROI转为灰度图，（高斯滤波，拉普拉斯变换？）

      - 灰度拉伸

        ![灰度拉伸](..\picture\灰度拉伸.png)

        https://blog.csdn.net/saltriver/article/details/79677199

      - 对比度增强

        ```c++
        Emphasize(ho_ImageScaleMax, &ho_ImageEmphasize, 7, 7, 1);
        ```

   3. 模板匹配：

      - 读取模板
      - 将现有的，处理过的ROI与带scale的模板进行匹配，获取到模板的ID（如果分数过低，则去除，<0.49）
      - 对获取到的模板进行仿射变换affine，有SRT（plot为绿色）
      - 对模板轮廓进行平移缩放处理，获得一个环形ROI

      - 对该使用deriche算法进行边缘轮廓亚像素提取

      - 2d的ICP，将模板匹配的结果往边缘检测的结果上ICP，计算transformation
      - 使用ICP再次对模板轮廓进行平移缩放处理，矫正（plot为红色）

      

#### 初始化三角化

1. 首先剔除匹配分数较低的观察，不需要进行三角化

2. 进行三角化

   - 2个view的三角化，采用中点法，INVERSE_DEPTH_WEIGHTED_MIDPOINT
   - N个view的三角化

   ```c++
           // 三个view及以上的三角化
           if (Ps.size() > 2) {
               TriangulateNView(xs, Ps, &X);
               //记录3D点值
               pair.X[0] = X.hnormalized()[0];
               pair.X[1] = X.hnormalized()[1];
               pair.X[2] = X.hnormalized()[2];
           }
           // 两个view的三角化
           else if (Ps.size() == 2) {
               Vec3 X_two;
               Triangulate2View(
                   Rs[0], Ts[0], Ks[0].inverse() * xs.col(0),
                   Rs[1], Ts[1], Ks[1].inverse() * xs.col(1),
                   X_two,
                   // 中点法
                   ETriangulationMethod::INVERSE_DEPTH_WEIGHTED_MIDPOINT);
               //记录3D点值
               pair.X[0] = X_two[0];
               pair.X[1] = X_two[1];
               pair.X[2] = X_two[2];
           }
   ```

3. 计算重投影误差，对于重投影误差过大的点，将其剔除（有可能是其2d检测有问题）。阈值为10



#### BA：优化3d点

1. 三角化

   - 2个view的三角化
   - N个view的三角化

2. 优化3d点

   进行BA优化：

   ```c++
       const bool bVerbose = true;
       const bool bMultithread = false;
       std::shared_ptr<Bundle_Adjustment_EB42X> ba_object =
           std::make_shared<Bundle_Adjustment_EB42X>(
               Bundle_Adjustment_EB42X::BA_EB42X_options(bVerbose, bMultithread));
   
   
       ba_object->AdjustIndividual(sfm_data,
           EB42X_Optimize_Options(
               EB42X_Structure_SRT_Parameter_Type::NONE,
               Intrinsic_Parameter_Type::NONE,
               Extrinsic_Parameter_Type::NONE,
               EB42X_Structure_Parameter_Type::ADJUST_ALL),
           // 大小圆
           std::unordered_set<uint32_t>(),
           // yhole
           std::unordered_set<uint32_t>(),
           // base_index
           std::map<uint32_t, uint32_t>(),
           // 网格点
           std::unordered_set<uint32_t>(),
           // 平面方程
           std::map<std::uint32_t, std::vector<double>>(),
           // landmark
           std::map<openMVG::IndexT, std::string>(),
           // big_small_circle_profile_z
           std::unordered_map<std::string, double>()
       );
   ```

   target：重投影误差

   参数：重建点的XYZ



#### 7Dof配准，计算RTS

1. 将重建点和三坐标测量值进行RTS配准

   ICP

   计算出三个参数：

   重建点到三坐标测量值的旋转R

   重建点到三坐标测量值的平移t

   重建点到三坐标测量值的缩放s

   这里用了openmvg提供的函数，没有使用PCL

   ```c++
       if (!use_pcl) {
           // Compute the Similarity transform
           //1.找到 重建点和三坐标测量值之间的srt-这是openmvg提供的函数。
           FindRTS(x2, x1, &params.Sc, &params.tc, &params.Rc);
           Refine_RTS(x2, x1, &params.Sc, &params.tc, &params.Rc);
   
           //2.找到 重建点和名义值之间的srt（sca,rot,tra）
           FindRTS(x2, x3, &params.Sc_norminal, &params.tc_norminal, &params.Rc_norminal);
           Refine_RTS(x2, x3, &params.Sc_norminal, &params.tc_norminal, &params.Rc_norminal);
       }
   ```

   

2. 将RTS应用到相机外参

   ```c++
       //将CAD坐标变换参数先叠加到相机内外参
       Mat3 R = params.Rc;
       double S = params.Sc;
       Vec3 t = params.tc;
       const openMVG::geometry::Similarity3 sim(openMVG::geometry::Pose3(R, -R.transpose() * t / S), S);
       openMVG::sfm::ApplySimilarity(sim, sfm_data_calibr);
   ```

   



#### BA：利用三坐标测量值优化RTS和相机外参

##### 添加一圈点：为基准孔添加一圈点，会增加N*36个landmark

1. 每个孔都会根据直径生成一圈点，但是以下的孔不会进行一圈点的生成：
   - 信息不全的孔，包括X、Y、Z以及D（基准孔X4X5除外）
   - 孔心不参与BA的不加入一圈点（螺纹孔）
2. 生成一圈点的过程是通过移动角度（360/N）



























## 测量

