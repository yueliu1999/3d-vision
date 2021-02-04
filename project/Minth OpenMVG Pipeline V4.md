## Minth OpenMVG Pipeline







[TOC]



<div style="page-break-after:always"></div>

## 1. 相机标定

```c++
void CalibrateAllCams(int times, bool use_kalibr, string curr_raw_impath) 
```

传参分别为标定次数，逻辑值（判断是否使用kalibr作为标定初值），以及当前原图路径。



#### 标定黑名单

​	根据.txt格式文档给出标定黑名单，在标定前加载roi过程中将不会加载位于该名单的孔。

```c++
EB42X_config::AppConfig& conf = EB42X_config::AppConfig::Instance(true);//true 代表标定
```

读取一些特殊的点集到参数结构中，以备后用：
    loadScrewed(params);//读取螺纹孔
    loadYHole(params);//读取侧立面孔
    loadCrossPoint(params);//读取激光线交点孔
    loadProfile(params);//读取大小圆list
    loadTHole(params);/读取/将通孔的下表面孔list

#### 初始内外参标定（OpenMVG代码不含）

- 使用kalibr进行相机标定，张正友标定法：

  标定

  - 内参：fx，fy，cx，cy
  - 外参：R，t
  - 畸变系数：
    - 径向畸变：k1，k2，k3
    - 切向畸变：p1，p2

  [张正友相机标定](https://zhuanlan.zhihu.com/p/94244568)

- 用colmap来进行重建、标定，基于marker （tbc）

- 读取初始内外参：

  - [ ] 读取txt文件

    利用AppConfig类调用方法AppConfig::loadCam 读取以.txt形式存储的初始内外参,路径为CAM_PATH

  - [ ] 读取yaml文件

    利用AppConfig类调用方法AppConfig::loadCamchainYaml()读取以.yaml形式存储的初始内外，路径为CAMCHAIN_YAML_PATH

​     

#### 模板生成

​	用名义值生成模板，将CAD模型的中心点和利用中心点和半径以及法向（大部分为水平孔，少部分侧面孔，少部分呈一定倾斜角度的孔，法向需区别）所生成的一圈点往各个相机下投影，由3维空间的圆投影为2d平面的椭圆。核心函数：

```c++
void getNorminalXYZ(MeasureParams& params, string path) //从json文件中读取名义值，存入参数结构中。名义值文件路径为EB42X_config::NORMINAL_XYZ_JSON_PATH。
```

```c++
void inline produce_models(MeasureParams& params, const string& ply_path, double scale)
```

​	传参分别为各孔心名义值，腰型孔（非圆形，会有单独的点云文件指示形状）点云文件路径，变换尺度。

​	若sfm_data中已经存储有相机参数，则可使用generate_models函数，否则需手动更改相机台数。

```c++
void inline generate_models(SfM_Data& sfm_data, MeasureParams& params, const string &ply_path, double scale)
```

​	

#### 2d检测

1. 图像去畸变，利用初始的畸变系数：k1，k2，k3，p1，p2

   其中

   - k1 k2 k3用于处理径向畸变
     $$
     x' = x(1+k_1r^2+k_2r^4+k_3r^6)
     \\
     y' = y(1+k_1r^2+k_2r^4+k_3r^6)
     $$
     
   - p1 p2用于处理切向畸变
     $$
     x' = x+(2p_1y+p_2(r^2+2x^2))
     \\
     y' = y+(p_1(r^2+2y^2)+2p_2x)
     $$
   
     核心函数：
   
   ```c++
   void undistortImages(string raw_img_path, string undist_img_path, EB42X_config::AppConfig& conf)
   ```
   
   ​	传参分别为原始图片路径，去畸变后图片路径，相机参数。内部调用OpenCV库函数
   
   ```c++
   cv::undistort(raw_img, undistort_img, I, dist_coeffs, noArray); //原图像，校正后图像，cameramatrix，失真系数，newcameramatrix
   ```
   
   
   
2. 利用Halcon/Linemod进行模板匹配或利用深度学习方法，找到圆孔的中心点：

   - [x] Halcon匹配

     ```c++
     int getPointPairs(string img_path, std::map<string, cv::Mat>& undistortImgMap, vector<Point2DPairs>& pointPairsList, bool cali)//图像路径，图像映射，输出匹配点对，判断是否为标定过程（该参数已弃用）
     ```

     在获取点对函数中调用模板匹配函数

     ```c++
     ShapeModel model = shape_matching_scale_icp(model_scale_path, cropGray, point, score, (double)scale); //此处进行模板匹配
     ```

     其中模板默认0.95-1.05的scale，有些孔位需要特殊的scale区间。如下，持续更新中

     ```c++
     
     	if (roiname.substr(0, 6) == "M82-FB" || roiname.substr(0, 6) == "M55-FB") {
     		maxscale = 1.25;
     	}
     	if (roiname.substr(0, 6) == "M41-DC") {
     		minscale = 1.05;
     		maxscale = 1.35;
     	}
     	if (roiname.substr(0, 6) == "M86-DC") {
     		maxscale = 1.35;
     	}
     	if (roiname.substr(0, 6) == "M60-DC") {
     		maxscale = 1.95;
     	}
     	if (roiname.substr(0, 6) == "M89-DC") {
     		minscale = 0.9;
     		maxscale = 1.1;
     	}
     	if (roiname.substr(0, 6) == "M88-DC") {
     		minscale = 0.95;
     		maxscale = 1.4;
     	}
     	if (roiname.substr(0, 5) == "M27-S") {
     		minscale = 0.95;
     		maxscale = 1.4;
     	}
     ```

     

     ```c++
     HalconCpp::FindScaledShapeModel(ho_ImageEmphasize, hv_ModelID, 0, 0, minscale, maxscale, 0.1, 1, 0.2, "least_squares_very_high",
     			0, 0.1, &hv_Row_check, &hv_Column_check, &hv_Angle_check, &hv_Scale, &hv_Score);
     
     //参照如下参数输入
     find_shape_model(Image : :  //搜索图像
                     ModelID, //模板句柄
                     AngleStart,  // 搜索时的起始角度
                     AngleExtent, //搜索时的角度范围，必须与创建模板时的有交集
                     MinScore, //最小匹配值，输出的匹配的得分Score 大于该值
                     NumMatches, //定义要输出的匹配的最大个数
                     MaxOverlap, //当找到的目标存在重叠时，且重叠大于该值时选择一个好的输出
     //如果MaxOverlap=0, 找到的目标区域不能存在重叠, 如果MaxOverla p=1，所有找到的目标区域都要返回。
                     SubPixel, //计算精度的设置，五种模式，多选2，3
                     NumLevels, //搜索时金字塔的层数
                     Greediness : //贪婪度，搜索启发式，一般都设为0.9
                     Row,Column, Angle, Score) //输出匹配位置的行和列坐标、角度、得分。
     ```

     贪婪度

     //如果Greediness=0，使用一个安全的搜索启发式，只要模板在图像中存在就一定能找到模板，然而这种方式下搜索是相对浪费时间的。如果Greediness=1，使用不安全的搜索启发式，这样即使模板存在于图像中，也有可能找不到模板，但只是少数情况。如果设置Greediness=0.9，在几乎所有的情况下，总能找到模型的匹配。

   

   ​		设定一个score阈值，如果模板匹配分数小于该值时则匹配失败。接下来进行一系列图形学变换。

   

   - [x] 深度学习孔心检测

     深度学习孔心检测的优势是，对于螺纹孔，光线较暗的水嘴孔等系列，几乎不会存在检测偏差较大的情况，而模板检测的算法，对于这类情况，可能会发生检测偏差较大或者漏检的情况。通过不同工件测量情况，对比替换深度学习检测结果是否对达标率有所提高，同样结合漏检率的情况，选择几个系列作为深度学习方法可替代系列，作为最终的孔心检测中心点。

     但是深度学习方法存在自身的劣势，包括检测的不够精准，存在椭圆构象偏差影响精度的情况，同时受训练集影响较大。

     结合模板匹配和深度学习的检测结果，可以把2d 圆心路径直接以文件形式保存，供下面函数调用。

     
     
     ```c++
     int getPointPairsFromText(string origin_path, string un_img_path, std::map<string, cv::Mat>& undistortImgMap, vector<Point2DPairs>& pointPairsList)//2d圆心路径，图片路径，图片映射，输出匹配点对
     ```
  
   - [ ] 混合方法

     当有些模板已有的2d圆心不存在匹配结果时，则换用混合方法，判断如果没有已匹配圆心，则对该图像进行模板匹配。

     
   
     ```c++
     if (pointfromtext.x <1e-12  && pointfromtext.y < 1e-12)//如果txt中没有该值的话，则用模板匹配得到的值{
       	ShapeModel model = shape_matching_scale_icp(model_scale_path, cropGray, point, score, (double)scale); //此处进行模板匹配
         if (point.x > 0 && point.y > 0 && point.x < crop.size().width / scale
         && point.y < crop.size().height / scale) {
              point.x += roiStruct.lx;
              point.y += roiStruct.ly;
              pairs.pointMap.emplace(ccdID, cv::Point2d(point.x, point.y));
              pairs.scoreMap.emplace(ccdID, score);
              Vec2 shift(roiStruct.lx, roiStruct.ly);
              pairs.contourMap.emplace(ccdID, model.ContoursTrans(shift, scale));
          }
  }
     ```

     

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
        $$
        I(x,y) = \frac{I(x,y)-I_{min}}{I_{max}-I_{min}}(MAX-MIN)+MIN
     $$
        

        [灰度拉伸](https://blog.csdn.net/saltriver/article/details/79677199)

      - 对比度增强
   
        ```c++
     Emphasize(ho_ImageScaleMax, &ho_ImageEmphasize, 7, 7, 1);
        ```

   3. Halcon模板匹配：
   
      - 读取模板
      - 将现有的，处理过的ROI与带scale的模板进行匹配，获取到模板的ID（如果分数过低，则去除，<0.49）
   - 对获取到的模板进行仿射变换affine，有SRT（plot为绿色）
      - 对模板轮廓进行平移缩放处理，获得一个环形ROI

      - 对该使用deriche算法进行边缘轮廓亚像素提取
   
   - 2d的ICP，将模板匹配的结果往边缘检测的结果上ICP，计算transformation
      - 使用ICP再次对模板轮廓进行平移缩放处理，矫正（plot为红色）

   4. Linemod模板匹配
   
      ```c++
      map<string, rad_err> linemod_match::match_final_calibra(string path)//linemod模板匹配函数,参数path为配置文件路径
      ```
      
         - 读取模板的训练文件，在生成模板图片之后，把每个view下的孔不同缩放尺度的图片信息提取。模板训练文件中保存不同尺度的模板的特征点。
      
         - 对ROI图片进行一系列的图片预处理操作，包括滤波，对比度增强，canny边缘检测等处理过程，这一步可以通过调节对比度参数和边缘检测的阈值，保留更多的边缘，从而大大降低漏检的情况，但是无效边缘过多，也会增加检测偏差过大的发生率。（Linemod也可以在原图上匹配，但是实践证明边缘图上效果更好。）
      
         - 将训练文件和边缘图片匹配，保留匹配分数最高分的case,阈值设为50分（这也是一个可以调节的参数，40分也是可以接受的），最高匹配分数高于匹配的阈值分数，则保留下这个最高匹配分数对应的匹配结果，最后将模板匹配的结果往边缘检测的结果上ICP，计算transformation，ICP之后对应模板的圆心点像素坐标作为最终的匹配结果。
      
           ```c++
            auto matches = detector.match(edge, 50, ids);//edge为边缘图像，50为阈值，ids是训练时指定的字符串，为"test"
           ```
      
         - 把最佳模板plot在ROI图片上，模板圆心坐标写到txt文件中。





#### 初始三角化

```c++
void triangulatePointPairs(vector<Point2DPairs>& pairList, MeasureParams& params, bool use_kalibr, int max_reproj_error = INT_MAX, double min_score_allow = 0)// 三角化点对，传参为：点对，螺纹孔参数，逻辑值（是否使用kalibr标定的内外参），重投影误差阈值，匹配分数阈值
```

1. 首先剔除匹配分数较低的观察，不需要进行三角化

   该方法与模板匹配的判断类似，可以在此处适当调高阈值，增加匹配精度要求。

2. 计算投影矩阵，如果标定阶段---使用kalibr的相机参数，如果测量阶段---使用标定优化后的相机参数。

    P_From_KRt(K, R, T, &P);  由内外参矩阵得到投影矩阵P

3. 进行三角化

   - 2个view的三角化，采用中点法，INVERSE_DEPTH_WEIGHTED_MIDPOINT
   - N个view的三角化

   code: 

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

4. 计算重投影误差，对于重投影误差过大的点，将其剔除（有可能是其2d检测有问题）。阈值为10，即重投影误差>10的view，则将其删除。若该孔只被两台相机看到，则不执行删除操作。

   ```c++
   for (int j = 0; j < Ps.size(); ++j) {
               int ccd_id = ccd_map[j];
               const Vec3 x_reprojected = Ps[j] * pair.X.homogeneous();
               const double error = (x_reprojected.hnormalized() - xs.col(j).hnormalized()).norm();
               //EXPECT_NEAR(error, 0.0, 1e-9);
               std::cout << "triangulate error: " << error << endl;
               file_w << ccd_id << "\t"
                   << error << endl;
               pair.errorMap.emplace(ccd_id, error); 
               curr_error = max(curr_error, error);
           }
           //剔除过大误差点
           if (curr_error > max_reproj_error) { //max_reproj_error=10
               it = pairList.erase(it);
               std::cout << "erase point: " << pair.roi_name << endl;
           }
           else {
               it++;
           }
   ```

   



#### BA：优化3d点

1. 三角化

   ```c++
   void triangulateStructurePoint(SfM_Data& sfm_data, MeasureParams& params, bool use_kalibr, int max_error = INT_MAX)// 三角化结构点，传参为：sfm_data，roi及index参数，逻辑值（是否使用kalibr标定的内外参），重投影误差阈值
   ```

   - 2个view的三角化

   - N个view的三角化

   - 再次进行针对重投影误差的筛选，剔除。

     - 以下为各剔除策略
       - [ ] 设置固定阈值，重投影误差高于阈值的全部删掉
       - [ ] 将每一个孔映射的所有2d观察根据重投影误差进行排序，删掉最大的一个或两个
       - [ ] 根据最后的测量结果，删掉对位置度有负面影响的2d观察
     - 以上策略均遵循共同的原则：如果#观察小于3，则不执行任何操作。

     code：

     ```c++
             for (int j = 0; j < Ps.size(); ++j) {
                 int ccd_id = ccd_map[j];
                 const Vec3 x_reprojected = Ps[j] * 				sfm_data.structure[feat_id].X.homogeneous();
                 const double error = (x_reprojected.hnormalized() - xs.col(j).hnormalized()).norm();
                 if (error > max_num)
                 {
                     max_num = error;
                     max_ccdid = ccd_id;
                 }
                 cout << "triangulate error: " << error << endl;
                 file_w << ccd_id << "\t"
                     << error << endl;
                 //删掉重建误差大的点        
                 if (error > max_error) {
                     if (structure.second.obs.size()> 3)
                     {                  
                         structure.second.obs.erase(ccd_id-1);//去掉大于阈值的view
                     }
                 }
             }
     ```

     

2. 优化3d点

   ```c++
   void refineTriangulatePoint(SfM_Data& sfm_data, MeasureParams& params)// 优化，传参为：sfm_data，roi及index参数
   ```

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
           // landmark
           std::map<openMVG::IndexT, std::string>(),
       );
   ```

   target:

   -  minimize reprojection error
     $$
     [x,y,1] = K[R|t][X,Y,Z,1]
     \\
      reprojection\ \ error = sqrt((x-x')^2+(y-y')^2)
     \\
     x' \ \  and \  \ y' \ \  is \  \ the \  \ point \ \ detected
     $$

   parameters:

   - 工件SRT：EB42X_Structure_SRT
   - 内参矩阵：Intrinsic_Parameter
   - 外参矩阵：Extrinsic_Parameter
   - 工件坐标XYZ：EB42X_Structure_Parameter



#### 7Dof(7 Degree of freedom)配准，计算RTS

```c++
void getCadRTS2Measure(SfM_Data& sfm_data, MeasureParams& params, bool use_pcl = false)// 获取名义值到测量值的SRT变换，传参为：sfm_data，roi及index参数，（pcl已弃用）
```

1. 将重建点和三坐标测量值进行RTS配准

   - 先算了一个两组点云的初始RTS，ICP算法，用Umeyama算法求解对应点关系矩阵

     [Umeyama in Eigen](https://blog.csdn.net/weixin_42823098/article/details/111308627)

   - 对两组点云进行细配准，用LM算法优化

   

   计算出三个参数：

   重建点到三坐标测量值的旋转R，自由度：3（表示为轴角形式，3*1向量）

   重建点到三坐标测量值的平移t，自由度：3

   重建点到三坐标测量值的缩放s，自由度：1

   

   ceres库中的旋转矩阵，四元数，轴角，欧拉角的各形式转换函数如下：

   ```c++
   ceres::AngleAxisToRotationMatrix(angleAxis, R_new.data());
   ceres::RotationMatrixToAngleAxis((const double*)R.data(), angleAxis);
   ```

   

   这里用了openmvg提供的函数，没有使用PCL

   ```c++
       if (!use_pcl) {
           // Compute the Similarity transform
           //1.找到 重建点和三坐标测量值之间的srt-这是openmvg提供的函数。
           FindRTS(x2, x1, &params.Sc, &params.tc, &params.Rc);//Xb=S*R*Xa+t 也即是重建值到测量值的变换
           Refine_RTS(x2, x1, &params.Sc, &params.tc, &params.Rc);
   
           //2.找到 重建点和名义值之间的srt（sca,rot,tra）
           FindRTS(x2, x3, &params.Sc_norminal, &params.tc_norminal, &params.Rc_norminal);//Xb=S*R*Xa+t 也即是重建值到名义值的变换
           Refine_RTS(x2, x3, &params.Sc_norminal, &params.tc_norminal, &params.Rc_norminal);
       }
   ```

   

2. 将RTS叠加到相机外参----//需要保持坐标系一致

   ```c++
   openMVG::geometry::Similarity3( const Pose3 & pose, const double scale );//给出sim函数，由pose和scale构成
   ```
   
   ```c++
   openMVG::sfm::ApplySimilarity(
     const geometry::Similarity3 & sim,
     SfM_Data & sfm_data,
    bool transform_priors = false
   );//应用similarity到sfm_data
   ```
   
   ```c++
       //将CAD坐标变换参数先叠加到相机内外参
       Mat3 R = params.Rc;
       double S = params.Sc;
       Vec3 t = params.tc;
       const openMVG::geometry::Similarity3 sim(openMVG::geometry::Pose3(R, -R.transpose() * t / S), S);
       openMVG::sfm::ApplySimilarity(sim, sfm_data_calibr);
   ```
   



#### BA：利用三坐标测量值优化相机外参

```c++
void calibrateCamWithMeasureXYZ(SfM_Data& sfm_data, vector<Point2DPairs>& pair_list, MeasureParams& params)//输入3d点，2d点以及必要参数
```

- 获取孔周轮廓点（均匀选取36个点），与圆心一起加入ba优化当中去，提高精度。

  ```c++
  void AddStructureDatumContour(SfM_Data& sfm_data, vector<Point2DPairs>& pairList, MeasureParams& params, bool cali)
  ```

  ```c++
  //被多台相机观测到的同一个点的轮廓的集合
          map<int, vector<Vec2>> ccd_contours = pair_base.contourMap;
          //添加每一个轮廓点
          for (int k = 0; k < X_3D.cols();k++) {
              Landmark landmark;
              landmark.X = X_3D.col(k);
              for (auto& c : ccd_contours) {
                  int ccd_id = c.first;
                  int cam_id = ccd_id - 1;
                  auto vec_points = c.second;
                  Vec2 pt = vec_points[k];
                  landmark.obs[cam_id] = Observation(pt, feat_id);
              }
              sfm_data.structure[feat_id] = landmark;
              string contour_name = roi_name + "_contour_" + to_string(k);
              params.landmarkIDs[contour_name] = feat_id;
              params.landmarkROIs[feat_id] = contour_name;
  
              //把yhole的contour也加到yhole里面
              if (params.yhole_set.count(roi_name) > 0) {
                  params.yhole_set.insert(contour_name);
              }
              feat_id++;
          }
  ```

  

- 遍历所有roi，生成一圈点

  ```c++
  bool get3DContourPoints(string roi_name, string type, PointCloudPtr& source_cloud, Mat3X& X_3D, const int sample_nums) 
  ```

  ```c++
   createTheoryCircle(normal_model,
              cv::Point3d(X_center[0], X_center[1], X_center[2]), radius * 1.0, source_cloud);
          X_3D.resize(3, source_cloud->points.size() / step);
          for (size_t j = 0; j < source_cloud->points.size(); ) {
              Vec3 cadpoint_3d(
                  double(source_cloud->points[j].x),
                  double(source_cloud->points[j].y),
                  double(source_cloud->points[j].z));
              X_3D.col(j / step) = cadpoint_3d;
              j += 1 * step;
          }
  ```

  

- 获取轮廓点index和孔心index的对应关系, 方便BA的时候把他们的Z/Y一起调整

  变量：roi_name & roi_name_base

  

- 利用三坐标的测量值作为孔心点的优化初值，遍历三坐标测量值，赋值孔心点的坐标

  ```c++
  sfm_data.structure[feat_id].X[0] = cad_point3D[0];
  sfm_data.structure[feat_id].X[1] = cad_point3D[1];
  sfm_data.structure[feat_id].X[2] = cad_point3D[2];//命名问题可能会产生混淆，注意cad_point3D在此处表示三坐标测量值，而不是cad名义值
  ```

​		

- 迭代优化 epoch = 5

  ```c++
  bool Bundle_Adjustment_EB42X::AdjustIndividual
       ba_object->AdjustIndividual(sfm_data,
              EB42X_Optimize_Options(
                  EB42X_Structure_SRT_Parameter_Type::NONE,
                  Intrinsic_Parameter_Type::NONE,//是否优化内参可选
                  Extrinsic_Parameter_Type::ADJUST_ALL,
                  EB42X_Structure_Parameter_Type::ADJUST_Z
                  //-> Use GCP to fit the SfM scene to the GT coordinates system (Scale, Rotation, Translation)
                  //Control_Point_Parameter(1, true)
              ),
              yholes,
              base_index//以及各种可选参数
          );
      
  ```

   	对于法向为[0, 0, 1]的孔，调整其Z坐标，对于法向[0, 1, 0]的孔，调整其Y坐标

   

   加入一些约束（可选）：

   - XYZ基准

     - [x] X4X5加权：

       50 * 重投影误差 or 10 \* 重投影误差

       因为XYZ基准的建系需要以X4为原点，X4X5为Y轴，所以给X4X5加权可以使建系更准

       ```c++
        if (roi_name == "X4Y6" || roi_name == "X5"){
                           const View* view = sfm_data.views.at(obs_it.first).get();
                           ceres::CostFunction* cost_function =
                               CameraCalibrateCostFunctionEB42X::Create(obs_it.second.x, 50.0);//给x4x5加权重，最后参数50处为weight，可根据需要修改权重
        }
       ```

          

     - [ ] 上下表面共同可见点约束（通孔约束）

       通过一些沟通上下表面的通孔深度一致性，约束两者的xy保持一致进行优化，delta-z为通孔深度，以此来添加一个强约束将上下表面联系起来（解决相机分布导致的上下表面几乎没有共同视野的问题）
       
       ```c++
       if (thole_index_set.count(feat_id) > 0){//如果是下表面孔
                           //内外参还是下表面孔，但是xyz是用上表面的值
                           string roi_up = th_set[roi_name];//找到对应的上表面孔
                           //找到上表面孔对应的featid
                           IndexT feat_idup = IDs[roi_up];
                           IndexT feat_idup_base = getBaseIndex(feat_idup, base_index);
                           double z = th_z_set[roi_name]/1000;//把对应的深度信息z 传进去
                           const View* view = sfm_data.views.at(obs_it.first).get();
                           double* x_3d, * y_3d, * z_3d;
                           x_3d = &map_x.at(feat_idup);
                           y_3d = &map_y.at(feat_idup);
                           z_3d = &map_z.at(feat_idup_base);
                           //新的creat函数for--thole
                           ceres::CostFunction* cost_function =
          CameraCalibrateCostFunctionEB42X4thole::Create(obs_it.second.x, z, x_3d[0], y_3d[0], z_3d[0],0.0);//给通孔加权重
       }
       ```
       
          
       
     - [ ] 下表面用上表面表示的约束：

       下表面的点用上表面的点来表示：
       $$
       X下=X上+delta(x下-x上)
       X下：下表面的点坐标
       X上：下表面对应上表面基准点的坐标
       x下：下面点cmm测量值
       x上：上面点cmm测量值
       $$
       为了对下表面的点进行优化,需要对上表面的点同时进行优化，即下表面和对应的上表面基准点公用一个变量进行优化。

       基准点的选择:上表面所有点中距离下表面最近的3个点；

       正则项约束：上表面基准点的变化范围应该很小,限制在cmm测量值附近很小的范围内波动。

       

   - 小基准

     - [x] 激光网格点约束：

       重投影误差（pixel）+网格点到三坐标拟合出来的平面方程的距离（mm）

       权重分配：1pixel=1mm

       因为

       - 小基准的建系需要用到激光网格点拟合的平面，加权则会使小基准的建立更准
       - 增加了特征点

       

     - [x] G1G2加权

       因为小基准的建系需要用到G1G2，所以给G1G2加权可以使建系更准

       

     - [x] X4X5加权

       实验得出x4x5加权可以使得建系结果更准，最终结论需要需要更多数据支持

       

- 大小圆

  - [x] 四个点Z均值约束：

    重投影误差 + abs（Z-Z'）

  - [x] 大小圆生成一圈点加Z均值约束

  - [ ] 单应性矩阵+平面方程约束





#### 保存更新后的内外参数

将优化后的三维点值更新到params.measureXYZ中去。对于Z孔，更新Z；对于Y孔，更新Y。

```c++
  //外参优化结束，更新测量的Z值
    for (auto& m : params.measureXYZ) {
        //Roi区域
        string roi_name = m.first;
        IndexT feat_id = params.landmarkIDs[roi_name];
        Vec3 old_v = m.second;
        //获取调整后的X
        Vec3 X = sfm_data.structure[feat_id].X;
        //进行坐标转换
        //Vec3 new_v = (1.0 / params.Sc_inv) * (params.Rc_inv.inverse() * (X - params.tc_inv)).transpose();

        cout << feat_id << endl;
        cout << roi_name << "_dx: " << old_v[0] - X[0] * 1000 << endl;
        cout << roi_name << "_dy: " << old_v[1] - X[1] * 1000 << endl;
        cout << roi_name << "_dz: " << old_v[2] - X[2] * 1000 << endl;

        //更新调整后的测量值
        m.second[0] = X[0] * 1000;
        m.second[1] = X[1] * 1000;
        m.second[2] = X[2] * 1000;
    }
```

并存储到，"measure_xyz.json“

注意：该值不会作为测量过程中的cmm值，而是作为一个中间变量输出观察优化效果。

接下来保存优化后的新内外参数

```c++
openMVG::sfm::Save(sfm_data_calibr, "./data/refined_params/SPEEDVISION_DATA.ply", openMVG::sfm::ALL);//点云文件，提供可视化验证
openMVG::sfm::Save(sfm_data_calibr, "./data/refined_params/SPEEDVISION_DATA.bin", openMVG::sfm::ALL);//.bin文件，提供测量步骤读取优化后的内外参
```



#### Other Ideas

- [x] 单task相机参数隔离

  各个task之间的相机参数单独优化，保存，重建，测量

  task

  - XYZ基准
  - 小基准
  - 大小圆

  

- [ ] 单孔相机参数隔离

  对于单个孔，都保存一份相机参数。对单个孔的相机参数进行单独的标定优化，保存，重建，测量。

  在标定优化A孔的时候，对A孔单独加权，加权尝试过两种方式，高斯加权效果会好一些

  - [ ] 仅对A孔加50倍权重，其他孔权重为1

  - [ ] 对A孔加50倍权重，其他孔权重取决于距A孔的距离，呈高斯分布，上下表面的孔权重不能传递


    $$
    Weight = W_{max}*e^{-\frac{distance^2}{\alpha^2r^2}}
    \\
    \alpha = 0.6,r = 0.3
    \\
    $$

    ```c++
    Vec3 other = params.measureXYZ[params.landmarkROIs[otherIndex]]/1000;
    Vec3 now= params.measureXYZ[params.landmarkROIs[index]]/1000;
    double distance = (other(0)-now(0))* (other(0) - now(0)) + (other(1) - now(1))* (other(1) - now(1));
    double weight;
    if (low != true) {
        weight = 50 * exp(-distance / ((0.6 * 0.6) * (0.1 * 0.1))) + 1;}
    else {
        weight = 35 * exp(-distance / ((0.6 * 0.6) * (0.1 * 0.1))) + 1;}
    ```

    

    测量：

    对于单个孔，读取保存的一份相机参数，单独重建单个孔。		

    ```c++
    for (auto sfm_data : sfm_data_measures) {   //逐点ba
        refineTriangulatePoint(sfm_data.second, params);
        Index feat_id = sfm_data.first;
        sfm_data_to_measure.structure[feat_id] = sfm_data.second.structure[feat_id];
    }
    ```

    

- [ ] 双工件联合标定

  Pipeline如上，输入为两个工件的2d匹配点，将两个工件的所有点加入到BA中进行标定优化

  

- [ ] 多工件移动联合标定

  Pipeline如上，输入为多个移动工件的2d匹配点，将两个工件的所有点加入到BA中进行标定优化

  将顶到头的工件称为初始工件，其他工件的重建点相对于初始工件的重建点有一个RT，将该RT加入到BA中进行优化

  

- [ ] 孔心点的uncentainty

  [蒙特卡洛方法](https://zh.wikipedia.org/wiki/%E8%92%99%E5%9C%B0%E5%8D%A1%E7%BE%85%E6%96%B9%E6%B3%95)

  在孔心的2d点上加入2d高斯随机噪声，用带噪声的点进行重建，观察其3d点距离三坐标测量值的3维点的距离有多远，以此来判断每个点的随机扰动不确定性。

  给该不确定性加以权重来进行标定测量

  

- [ ] 跨工件时模板更新

  前提：认为跨工件场景下，工件位姿前后会存在RT。

  基于此，为更好的匹配，需要生成更具有针对性的模板。因此计算出两个工件之间的相对位姿RT之后，将该RT加入到生成模板过程中，控制模板法向。

  

<div style="page-break-after:always"></div>

## 2. 工件测量

#### 测量白名单

​	根据.txt格式文档给出测量白名单，在标定前加载roi过程中将仅加载位于该名单的孔。

```c++
EB42X_config::AppConfig& conf = EB42X_config::AppConfig::Instance(false);//false代表测量
```



#### 2d检测（流程与标定过程相同）

1. 图像去畸变，利用初始的畸变系数：k1，k2，k3，p1，p2

2. 利用halcon进行模板匹配：

   1. ROI：获取到每个ROI（手工标注的ROI）

      格式：camera-CCDn.json代表CCDn相机view下的ROI

      - name：测点名字
      - lx：左上角点的x
      - ly：左上角点的y
      - rx：右下角点的x
      - ry：右下角点的y

      标注 / 检查ROI软件link：

      [Minth ROI Labelme Web](http://label.bslapp.me/#/)

      

   2. 图像预处理：

      - 将ROI转为灰度图，（高斯滤波，拉普拉斯变换？）

      - 灰度拉伸

        
        $$
        I(x,y) = \frac{I(x,y)-I_{min}}{I_{max}-I_{min}}(MAX-MIN)+MIN
      $$
        

        [灰度拉伸](https://blog.csdn.net/saltriver/article/details/79677199)
   
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




#### 三角化

利用优化后的外参，重新三角化全部点

- 2个views的三角化

  - [ ] DLT(Direct Linear Transform)
  - [ ] L1_Angular
  - [ ] LInfinity_Angular
  - [x] Inverse_depth_weighted_midpoint

- N个views的三角化

  - [x] DLT(Direct Linear Transform)

    拿着3d点的坐标XYZ，往各个view下投影，得到2d坐标，列出方程组

    SVD求(超定)方程的解，得到最小二乘解

  - [ ] RANSAC(Random Sample Consensus)
    - 先从所有view中随机选2个view进行三角化，将该重建点往各个view下投，计算重投影误差，如果重投影误差小于一定阈值，则inliers+1，多做几组，从中选出inliers数量最大的那组inliers
    - 拿inliers进行三角化，SVD

  

#### 建系

- 全局坐标建系

  1. 平面法向默认垂直向上，即沿用名义值坐标系z轴，
  2. X4为原点，X4到X5的连线到平面的投影为y轴，
  3. （0,0,1）为z轴，x轴为y叉乘z，建系完成，
  4. 将应在全局基准下测量的重建点变换到该坐标系下进行测量。

  核心函数：

  ```c++
  void convertToDatum(SfM_Data& sfm_data_to_measure, MeasureParams& params)//X4X5建系
      
      //1.平移对准X4的名义值
      //2.旋转X5
      //先平移到相对X4的坐标系
      //再旋转
      //旋转完再平移回到x4名义值
  ```

  

- 小基准建系（tbc）

  - [ ] NN小基准

    建系方式基本与全局坐标相同

    1. 选取小基准平面上的若干特征点（至少四个）拟合一个平面NN，得到平面法向，
    2. M87-DC-B27为原点，M87-DC-B27到FF2的连线到NN平面的投影为y轴，
    3. 平面法向为z轴，x轴为y叉乘z，建系完成，
    4. 将应在该基准下测量的重建点变换到该坐标系下进行测量。

  - [ ] （tbc）

  

#### 计算

- 位置度

  计算公式：
  $$
  \left\{
  \begin{align}
  
  position = 2*sqrt((x_m-x_N)^2+(y_m-y_N)^2) \quad if\quad norm = [0,0,1]
  \\
  position = 2*sqrt((y_m-y_N)^2+(z_m-z_N)^2) \quad if\quad norm = [0,1,0]
  \end{align}
  \right.
  $$
  

  - XYZ基准的位置度
  - 小基准的位置度

- 平面度
  $$
  flatness = 正差+负差
  $$
  
- 轮廓度
  $$
  profile = abs(distance_{measure}-distance_{norminal})
  $$
  

  - 平面的轮廓度
  - 大小圆的轮廓度



#### 对标三坐标测量值

1. 三坐标测量值提取

   - excel->json
   - txt->json

   将测点号的position，x，y，z，flatness，profile作为value，将测点号作为key

2. 三坐标重复性分析

   多组三坐标测量值进行分析，计算两次的三坐标的差值，得到以下分析结果

   - count
   - mean
   - std
   - min
   - <25%
   - <50%
   - <75%
   - max
   - hist plot

3. 对标

   计算三坐标测量值position和本系统的重建值position的差值

   统计

   - <0.2百分比
   - <0.3百分比
   - <0.4百分比
   
   统计上x和y轴向坐标的相关性
   
   - <0.15百分比
   - <0.2百分比
   - <0.3百分比
