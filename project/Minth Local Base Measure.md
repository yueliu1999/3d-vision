## Minth Local Base Measure Pipeline

## 相机标定

参考OpenMVG pipeline



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

利用优化后的外参，重新三角化全部点

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

















