# Homography

单应性原理被广泛应用于图像配准，全景拼接，SLAM，AR领域

## 图像变换和坐标系的关系

- 旋转rotation

  ![img](https://pic3.zhimg.com/80/v2-d7ae1b0b17c5889aed8f2d2f26995cca_720w.jpg)

  <img src="https://pic4.zhimg.com/80/v2-12ea7ce80faf4bd643040de97b9162b3_720w.jpg" alt="img" style="zoom:50%;" />

  ![img](https://pic4.zhimg.com/80/v2-6f692bcae7b21f2e218eb5a0340b9c47_720w.jpg)

  改变了：

  - 位置

  保留了：

  - 形状
  - 大小

  

  

- 平移translation

  ![img](https://pic4.zhimg.com/80/v2-71f3b237e19d7bb6f88b217408a07547_720w.jpg)

  <img src="https://pic2.zhimg.com/80/v2-a3645edc2ccbeefa5f3c8fd795262889_720w.jpg" alt="img" style="zoom:50%;" />

  ![img](https://pic1.zhimg.com/80/v2-70c24299b19575fdb91c3c7b0007af4c_720w.jpg)

  引入齐次坐标：

  ![img](https://pic4.zhimg.com/80/v2-3115dafb07613e89c36a1a34b220c083_720w.jpg)

  旋转矩阵是正交矩阵

  R^TR = RR^T = I

  <img src="https://pic2.zhimg.com/80/v2-654e0de7d66bebec4fd2435994a66b09_720w.jpg" alt="img" style="zoom:50%;" />

  改变了：

  - 位置

  保留了：

  - 形状
  - 大小

  



- 仿射affine

  ![img](https://pic1.zhimg.com/80/v2-4608508f42b3f013598a4bae5632a874_720w.jpg)

  旋转矩阵由1个自由度，变为4个自由度

  A可以为任意的2*2的矩阵，与R一定为正交矩阵，且行列式为1不同

  <img src="https://pic3.zhimg.com/80/v2-648a9b90ba8e0100b02d2acdef3b10a2_720w.jpg" alt="img" style="zoom:50%;" />

  仿射变换：

  改变了：

  - 形状
  - 位置
  - 大小

  保留了：

  - 平行性
  - 直线性

  

  ![img](https://pic2.zhimg.com/80/v2-1e174d6612e00f7ab85365788ae61d05_720w.jpg)

  

  

  

- 投影变换（单应性变换）homograph

  ![img](https://pic2.zhimg.com/80/v2-2ba1dab8ea5d717400bb9ef42219e749_720w.jpg)

  <img src="https://pic4.zhimg.com/80/v2-717559e850a3a981043e3a809c512073_720w.jpg" alt="img" style="zoom:50%;" />

  改变了：

  - 位置
  - 形状
  - 大小
  - 平直性



总结一下：

1. 刚体变换：平移+旋转，只改变物体位置，不改变物体形状
2. 相似变换：平移+旋转+放缩，不改变物体形状
3. 仿射变换：改变物体位置和形状，但是保持平直性
4. 投影变换：彻底改变物体位置和形状



<img src="https://pic3.zhimg.com/80/v2-8de7efe39d587b3957a193c8b1729f46_720w.jpg" alt="img" style="zoom:70%;" />



H = 

[A2*2 ，   T2\*1

V^T    ，      s   ]

![img](https://pic2.zhimg.com/80/v2-6b65de3bf3cabbe55a5e7d2b43018939_720w.jpg)



其中



- **A2*2**代表**仿射变换**参数

- **T2\*1**代表**平移变换**参数

- V^T = [v1,v2]表示一种"变换后边缘交点"关系
- s则是一个与V^T相关的缩放因子

HX1 = X2，一般会归一化为1

![img](https://pic3.zhimg.com/80/v2-f3e58c80ff64f93ec72c0f96ddb6c8fe_720w.jpg)





## 平面坐标系与齐次坐标系

w = 0的时候表示无穷远点



## 单应性变换

单应性变换是2d到2d的变换

单应性矩阵是一个3*3的矩阵

3*3   3\*1   3\*1

![img](https://pic3.zhimg.com/80/v2-75d5a3d1f42c58091aee60865e1984d6_720w.jpg)

![img](https://pic4.zhimg.com/80/v2-cfd8fd926ab531ee68a068d964eaf267_720w.jpg)







## 单应性变换的求解

通过

Hx = x‘构建方程组

AX = 0的形式，其中X是H拉成一列，A和x有关



求解方程组即可，一对点可以提供两个方程，而单应性矩阵具有8个自由度，可以通过最少四个点来计算出其唯一解。



可以加入约束：

- H33 = 1
- 或sqrt(ΣHij) = 1





找出四个点来计算homography

python代码

```python
import cv2
import numpy as np

im1 = cv2.imread('E:/Desktop/3d-vision/picture/homography_left.png')
im2 = cv2.imread('E:/Desktop/3d-vision/picture/homography_right.png')

src_points = np.array([[564, 375], [1235, 306], [635, 825], [1255, 668]])
dst_points = np.array([[754, 319], [1454, 371], [726, 730], [1366, 933]])

H, _ = cv2.findHomography(src_points, dst_points)

h, w = im2.shape[:2]

im1_warp = cv2.warpPerspective(im1, H, (w, h))
img_mix = cv2.addWeighted(im1_warp, 0.1, im2, 0.1, 0)
cv2.imwrite("./homography.png",im1_warp)
cv2.imwrite("./concatenate.png",img_mix)

# cv2.imshow("1",im2_warp)
# cv2.waitKey(0)
```

left：

<img src="..\picture\homography_left.png" alt="homography_left" style="zoom:25%;" />

right：

<img src="..\picture\homography_right.png" alt="homography_right" style="zoom:25%;" />

left->H->homograpy：

<img src="..\picture\homography.png" alt="homography" style="zoom:25%;" />



left & right：

<img src="..\picture\concatenate.png" alt="concatenate" style="zoom:25%;" />



可以看到电脑平面是一个平面，所以通过一个H从2d到2d可以重合上，而其他的物体，例如杯子，不在一个平面上，所以









手写部分

![homography_script](..\picture\homography_script.png)



## 传统方法估计单应性矩阵

步骤：

1. 提取每张图片的SIFT/SURF/FAST/ORB等特征点
2. 计算每个特征点所对应的描述子
3. 通过匹配特征点描述子，找到两种图重匹配的特征点对，汉明距离？
4. 用RANSAC算法剔除错误匹配
5. 求解方程组，计算Homography单应性变换矩阵，SVD？



python代码

```python
#coding:utf-8

# This code only tested in OpenCV 3.4.2!
import cv2 
import numpy as np

# 读取图片
im1 = cv2.imread('left.jpg')
im2 = cv2.imread('right.jpg')

# 计算SURF特征点和对应的描述子，kp存储特征点坐标，des存储对应描述子
surf = cv2.xfeatures2d.SURF_create()
kp1, des1 = surf.detectAndCompute(im1, None)
kp2, des2 = surf.detectAndCompute(im2, None)

# 匹配特征点描述子
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 提取匹配较好的特征点
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

# 通过特征点坐标计算单应性矩阵H
# （findHomography中使用了RANSAC算法剔初错误匹配）
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

# 使用单应性矩阵计算变换结果并绘图
h, w, d = im1.shape
pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts, H)
img2 = cv2.polylines(im2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

im3 = cv2.drawMatches(im1, kp1, im2, kp2, good, None, **draw_params)
```



特征点匹配的数量远远大于4，此时需要解超定方程组（类似于求解线性回归）



超定方程组：

方程个数大于未知量个数的方程组，一般精确解，如果有精确解，则称为是相容的

一般采用最小二乘法求解，在无法完全满足给定的浙西条件的情况下，求一个最接近的解

如果有向量a使得值最小，则称a为超定方程组的最小二乘解







## 深度学习在单应性方向的进展

HomographyNet



是2016年CVPR的一种用深度学习计算单应性矩阵的变换网络：

输入两种图片，直接输出单应性矩阵H

![img](https://pic2.zhimg.com/80/v2-59181d31895c04627535996f6ec37ba1_720w.jpg)

可以进行：

- 回归任务，直接回归float values
- 分类任务，具有confidence，强行进行区域划分，会产生一定的误差，但是能够输出分类置信度评价当前效果好坏，更便于实际应用。





数据生成方式：

1. 随机取一个patchA
2. 对矩形进行扰动，获取4个（deltax，deltay）
3. 再通过4组（deltax，deltay）计算HAB
4. 最后将图像通过HBA变换，在变换后图像p位置获取正方形图像块Patch B

![img](https://pic2.zhimg.com/80/v2-6a1d68c917423232d44143a9242aebb1_720w.jpg)



![img](https://pic2.zhimg.com/80/v2-42772ffbd19482a8e77d164eeccb98c1_720w.jpg)



深度学习提取特征点与描述子

- DELF：DEept Local Features









reference：

https://zhuanlan.zhihu.com/p/74597564

https://baike.baidu.com/item/%E8%B6%85%E5%AE%9A%E6%96%B9%E7%A8%8B%E7%BB%84