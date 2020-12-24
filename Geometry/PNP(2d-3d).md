# PNP(2d-3d)

视觉里程计的任务是估算相邻图片之间的运动，以及局部图片的样子。

三种方法：

- 对极几何（2d-2d）

- PNP（2d-3d）
- 三角测量 ICP（3d-3d）





回顾：

## 2d-2d对极约束

本质矩阵E（Essential Matrix）

基础矩阵F（Fundamental Matrix）

单应性矩阵H（Homography）



E = t^R

F = (K^-1)^TEK-1

x2^T E x2 = p2^T F p1 = 0

pipeline：

1. 根据匹配点的像素位置求出E或者F
2. 根据E或者F求出R和t



## 三角测量

- 2个view的三角化
- N个view的三角化



## 3d-2d: PNP（perspective-n-point）

当知道了n个3d空间位置以及投影位置时，如何估计相机的位姿

特征点的3d位置可以由

- 三角化
- RGB-D相机的深度图

来确定



在双目或者RGB-D的视觉里程计中，我们可以直接使用PnP估计相机的运动

在单目视觉里程计中，必须先初始化，然后才能使用PnP



3d-2d方法**不需要使用对极约束**，又可以在**很少的匹配点**中获得较好的运动估计。

PnP问题：

- 对3对点估计位姿P3P
- 直接线性变换DLT
- EPnP
- 非线性优化BA





PnP的求解方法：

1. 直接线性变换DLT

   对于空间点P，假设齐次坐标为P = （X，Y，Z，1）^T，在图像中投影得到的特征点坐标为x1 = （u1，v1，1）^T，已知P和X1，求解R和T

   sx = [R|t]P

   1个特征点（2d、3d）可以提供两个方程

   t1^TP-t3^TPu1 = 0

   t2^TP-t3^TPv1 = 0

   R和t一共有12个未知量，所以需要12个方程，一共需要6对3d-2d点

   ![pnp_DLT](..\picture\pnp_DLT.png)

   

2. P3P

   P3P只需要3对匹配点，相对于DLT的6对匹配点

   设ABC是三个世界坐标系中的点，而abc为ABC投影在图像上的点

   ![img](https://pic2.zhimg.com/80/v2-4fdf2f8a1366473064c55386e4f0322d_720w.jpg)

   可以找到3个相似三角形

   oab和oAB

   oac和oAC

   obc和oBC

   根据余弦定理：

   推导方程，求出3d点，转为3d-3d的问题，求解Rt

   缺点：

   - 只利用了3个点的信息，难以利用更多信息
   - 如果存在误匹配，则整个计算失效

   ![pnp_P3P](..\picture\pnp_P3P.png)

   

3. 非线性优化求解 Bundle Adjustment

   思想：最小化重投影误差，求解位姿RT

   ![img](https://pic3.zhimg.com/80/v2-5ced963c626a6a5755c63183f656feee_720w.jpg)

   

   

   











## 3d-3d: ICP（Iterative Closest Point，迭代最近点）

假设有一组匹配好的3d点，可以估计相机的位姿

ICP位姿估计没有出现相机模型，和相机没有关系

在RGBD SLAM中，可以用这种方式估计相机位姿

求解方式：

- 线性方法：SVD
- 非线性方法：BA







reference:

https://zhuanlan.zhihu.com/p/80921759

https://zhuanlan.zhihu.com/p/61742217



