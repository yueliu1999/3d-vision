## Minth Local Base Measure Pipeline

## 相机标定

参考OpenMVG pipeline

- [x] 加入激光网格点约束
- [x] G1G2加权
- [x] X4X5加权



## 小基准测量

#### 2d检测

1. 图像去畸变，包括：

   - 径向畸变

     k1 k2 k3

   - 切向畸变

     p1 p2

     

2. 检测：

   - 孔心点

     1. 用名义值CAD往各个相机投影，生成模板
     2. ROI
     3. 图像预处理
     4. 用生成好的模板进行模板匹配，得到孔心点

     用Halcon或Linemod进行模板匹配或深度学习来检测孔心点

     参考OpenMVG pipeline

   - 激光线

     1. ROI：获取到每个ROI（手工标注的ROI）

        格式：camera-CCDn.json代表CCDn相机view下的ROI

        - name：测点名字
        - lx：左上角点的x
        - ly：左上角点的y
        - rx：右下角点的x
        - ry：右下角点的y

        标注 / 检查ROI软件link：

        [Minth ROI Labelme Web](http://label.bslapp.me/#/)

        

     2. 图像预处理

        先做一个闭运算，先膨胀后腐蚀，使得激光线能够更完整一些

        [开运算和闭运算](https://zhuanlan.zhihu.com/p/46306138)

        闭运算可以

        1. 填平小孔，弥合小裂缝，而总的位置和形状不变
        2. 闭运算是通过填充图像的凹角来滤波图像的
        3. 结构元素大小的不同将导致滤波效果的不同
        4. 不同结构元素的选择导致了不同的分割

        

     3. 提取激光线中心算法，[steger](https://blog.csdn.net/Dangkie/article/details/78996761)：

        1. GaussBlur

           其中高斯方差需要
           
           
           
           
           $$
           \delta<\frac{w}{\sqrt3}
           \\ w为光条宽度
           $$

        2. Hessian矩阵**最大特征值**对应的特征向量对应于光条的法线方向
           $$
           (n_x,\ n_y)
           $$
           以点
           $$
           (x_0,\ y_0)
           $$
           为基准点，则光条中心的亚像素坐标为
           $$
           (p_x, \ p_y) = (x_0+tn_x,\ y_0+tn_y)
           $$
           其中t的计算公式为
           $$
           t = -\frac{n_xr_x+n_yr_y}{n_x^2r_{xx}+2n_xn_yr_{xy}+n_y^2r_{yy}}
           $$
           如果满足以下条件，
           $$
           (tn_x, \ tn_y)\in [-0.5, \ 0.5] \times [-0.5 \ 0.5]
           $$
           即**一阶导数为零的点位于当前像素内**，且**二阶导数大于指定的阈值**

           则该点
           $$
           (x_0,\ y_0)
           $$
           为光条的中心点，点
           $$
           (p_x, \ p_y)
           $$
           为亚像素坐标



#### 三角化

利用优化后的外参，重新三角化全部点，包括

1. 孔心点
2. 网格点



三角化的过程：

- 2个views的三角化

  - [ ]  DLT(Direct Linear Transform)
  - [ ]  L1_Angular
  - [ ]  LInfinity_Angular
  - [x]  Inverse_depth_weighted_midpoint

- N个views的三角化

  - [x]  DLT(Direct Linear Transform) 拿着3d点的坐标XYZ，往各个view下投影，得到2d坐标，列出方程组 SVD求(超定)方程的解，得到最小二乘解

  - [ ] RANSAC(Random Sample Consensus)
    - 先从所有view中随机选2个view进行三角化，将该重建点往各个view下投，计算重投影误差，如果重投影误差小于一定阈值，则inliers+1，多做几组，从中选出inliers数量最大的那组inliers
    - 拿inliers进行三角化，SVD





#### BA：优化3d点

进行BA优化

target:

- reprojection error
  $$
  [x,y,1] = K[R|t][X,Y,Z,1]
  \\
   reprojection\ \ error = sqrt((x-x')^2+(y-y')^2)
  \\
  x' \ \  and \  \ y' \ \  is \  \ the \  \ point \ \ detected
  $$

parameters:

- X
- Y
- Z



#### 平面拟合

对于重建的网格点，需要对之进行平面拟合，需要拟合的平面包括

- PM01
- PM02
- PM03
- PM04
- PM05
- PM06
- PM28
- PM29
- PM30
- PM31
- PM32
- PM33
- PM37-A-FR
- PM37-A-RR
- PM37-B-FR
- PM37-B-RR
- PM37-C-FR
- PM37-C-RR
- PM37-D-FR
- PM37-D-RR
- PM37-E-FR
- PM37-E-RR
- AA-FR
- AA-RR



平面拟合流程

1. RANSAC(Random sample consensus)

   从所有点中随机选取3个点进行平面方程的计算，计算其他各个点到该平面的距离，如果距离小于一定阈值，则inliers+1，多做几组，从中选出inliers数量最大的那组inliers

2. 最小二乘

   用选出的inliers进行最小二乘拟合平面





#### 建立小基准

利用以上拟合的平面以及G1G2进行小基准坐标系的建立



#### 测量

将圆孔孔心重建的坐标转换到小基准坐标系下进行计算

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

- 平面度
  $$
  flatness = 正差+负差
  $$
  

- 面轮廓度
  $$
  
  profile = abs(distance_{measure}-distance_{norminal})
  $$



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

   计算三坐标测量值和我们的测量值的差值

   统计

   - <0.2百分比
   - <0.3百分比
   - <0.4百分比









