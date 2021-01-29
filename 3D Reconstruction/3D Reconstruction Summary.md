## 3D Reconstruction Summary

### 基于传统多视图几何的三维重建算法

按照传感器是否主动向物体照射光源分为

- 主动式：通过传感器主动地向物体照射信号，然后依靠解析返回的信号来获得物体的三维信息

  - 结构光
  - TOF激光飞行时间
  - 三角测距法

  

- 被动式：直接依靠周围环境光源来获取RGB图像，通过多视图几何原理对图像进行解析，从而获取物体的三维信息

  - 单目视觉

    - 离线重建

      运动恢复法SfM

    - 在线重建

      - 渐进式
        - REMODE
        - SVO
      - 直接式

  - 双目/多目视觉

    难点在于左右相机图片的匹配

    - 全局匹配GM
    - 本全局匹配SGBM
    - 局部匹配BM

    

    

- 消费级RGB-D相机进行三维重建，例如基于微软的Kinect V1产品

  - Kinect Fusion
  - Dynamic Fusion
  - Bundle Fusion
  - Fusion 4D
  - Volumn









### 基于深度学习的三维重建算法

CNN探索三维重建

- 为传统重建算法性能优化提供新思路

  - DeepVO

    基于深度递归卷积神经网络RCNN从RGB视频中推断出姿态，不采用传统视觉里程计中的任何模块

  - BA-Net

    将SfM算法中的BA优化作为神经网络的一层，以便训练出更好的基函数生成网络

  - Code SLAM

    其通过神经网络提取出若干个基函数表示场景的深度

- 将深度学习重建算法和传统三维重建算法进行融合，优势互补

  CNN-SLAM13将CNN预测的致密深度图和单目SLAM的结果进行融合，在单目SLAM接近失败图像位置如低纹理区域，其融合方法给予更多权重于深度学习方案，提高了重建的效果

  

- 模仿动物视觉，直接利用深度学习算法进行三维重建

  - 深度图depth map，2d图片，每个像素从视点到物体的距离，以灰度图表示，越近越黑

  - 体素voxel，和像素的概念类似，3d的像素

    **Depth Map Prediction from a Single Image using a Multi-Scale Deep Network, 2014**

    DL做三维重建的开山之作

    直接用单张图片使用神经网络直接恢复深度图方法，将网格分为全局粗配准和局部精估计，使用一个尺度不变的损失函数进行回归

    **3D-R2N2: A unified approach for single and multi-view 3d object reconstruction, 2016**

    

  - 点云，每个点都有XYZ坐标，乃至色彩、反射强度、法向等信息

    点云更为简单，统一的结构，容易学习

    点云中缺少连接性，缺乏表面信息

     **A Point Set Generation Network for 3D Object Reconstruction From a Single Image, 2017**

    **Point-Based Multi-View Stereo Network, 2019**

    融合三维深度和二维纹理信息，提高点云重建精度

    

  - 网格，即多边形网格，容易计算

    - 基于体素，计算量大。并且分辨率和精度难平衡
    - 基于点云，点云之间缺少连接性，重建后物体表面不光滑

    网格的表示方法，轻量、形状细节丰富的特点，重要的是相邻点之间有连接关系

    网格由顶点、边、面来描述3D物体，这正好对应于图卷积神经网络的M = （V，E，F）

    Pixel2Mesh，用三角网格来做单张RGB图像的三维重建

    - 对任意的输入图像都初始化一个椭球体作为三维形状
    - 网格分为两部分
      - 一部分用全卷积神经网络提取图像特征
      - 另一部分用图卷积神经网络表示三维结构
    - 对三维网格不断进行变形，最终输出物体的形状

    

    













reference:

[基于深度学习的三维重建算法综述](https://zhuanlan.zhihu.com/p/108198728)

