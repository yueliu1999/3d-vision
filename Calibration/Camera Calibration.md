## Camera Calibration

相机标定涉及到的知识面：

- 成像几何

  世界坐标系->相机坐标系->成像平面坐标系->像素坐标系

- 镜头畸变

  - 径向畸变：镜头引起
  - 切向畸变：感光元器件和镜头不平行

- 单应性矩阵：平面到平面的transformation，从2d点（3\*1）到2d点，为3*3的矩阵

- 非线性优化：BA优化



相机标定的分类：

- 按照特征点分类：
  - 自标定

    找图片中的特征点，sift特征，surf特征

    

  - 标定板标定

    特征点易求，稳定性好

- 按照相机是否静止分类：

  - 静态相机标定

    标定板运动，相机静止

  - 动态相机标定

    标定板静止，相机运动

    有点类似于SLAM或3d重建

    

## 1物理模型

1. **浅谈标定**

   普通相机成像的误差

   - sensor制造产生的误差

     - pixel不是正方形
     - sensor歪斜

   - 镜头制造和安装产生的误差

     - 径向畸变：

       镜头一般存在非线性镜像畸变，包括：

       - 桶形畸变
       - 枕形畸变

     - 切向畸变：

       镜头与相机sensor安装不平行，还会产生切向畸变

   

2. **透视投影（Perspective）模型**

   ![img](https://pic4.zhimg.com/80/v2-f06178480b426e54a55cc8ee606116b7_720w.jpg)

   ![img](https://pic2.zhimg.com/80/v2-f65bac8057180048bde53918904e9955_720w.jpg)

   ![pinhole_camera_model](..\Picture\pinhole_camera_model.png)

   

3. **主点偏移**

   成像平面坐标系x0y和像素坐标系u0v的原点不重合，像素坐标系往往在图片的左上角，而光轴经过图像中心，因此图像坐标系和相机坐标系不重合。两个坐标系之间存在一个平移运动

   cx和cy

   ![img](https://pic3.zhimg.com/80/v2-849363945c1864677a7e4356806788b2_720w.jpg)

   

4. **图像传感器特性**

   sensors在制造的过程中可能不是正方形的，同时可能存在歪斜（skewed），因此需要考虑这些因素，传感器歪斜和不是正方形主要是对相机x和y方向的焦距产生影响

   ![image_sensors](..\Picture\image_sensors.png)

   ![img](https://pic2.zhimg.com/80/v2-e58c0b2bdcf3ae97497418bb86cfa501_720w.jpg)

   $\lambda$是缩放系数，在3d点到2d点的投影的过程中，可以直接将uvw的w归一化，从而求得u和v

   而2d无法求3d，因为丢失深度信息

   K是相机内参Intrinsic parameters

   

5. **镜头畸变对成像的影响**

   pinhole camera model充分考虑相机内参对成像的影响，但是没有考虑另一个重要的部分：**镜头**。

   镜头分为

   - 普通镜头
   - 广角镜头
   - 鱼眼镜头

   

   普通镜头主要考虑

   - 镜像畸变
   - 切向畸变

   都可以用多项式来近似

   不适用于大广角、鱼眼镜头

   

   1. **径向畸变**

      ![径向畸变](..\Picture\径向畸变.png)

      从左到右为：

      - 无畸变
      - 桶形畸变
      - 枕形畸变

      径向畸变以某个中心点向外延伸，且越往外，畸变越大，显然畸变和距离是一种**非线性的变换关系**，可以通过**多项式来近似**
   
      
      $$
      \begin{cases}
      x_{corrected} = x(1+k_1r^2+k_2r^4+k_3r^6)
      \\
      y_{corrected} = x(1+k_1r^2+k_2r^4+k_3r^6)
      \end{cases}
   \\
      其中x_{corrected}是去畸变后的x坐标，x是去畸变前的x坐标
   \\
      k1,k2,k3是径向畸变系数
   \\
      r = x^2+y^2
   $$
   
      
   
      
   
   2. **切向畸变** tangential distortion
   
      主要发生在**相机sensor**和**镜头不平行**的情况下
   
      因为有夹角，所以光透过镜头传到图像传感器上时，成像位置发生了变化
   
      ![tangential_distortion](..\Picture\tangential_distortion.png)
      $$
      \begin{cases}
      x_{corrected} = x + [2p_1xy+p_2(r^2+2x^2)]
      \\
      y_{corrected} = y + [2p_2xy+p_1(r^2+2x^2)]
      \end{cases}
      \\
      p1,p2是切向畸变的系数
      \\
      r^2 = x^2+y^2
      $$
      
   
      
      $$
      一共有5个畸变系数
      \\
      \\
      径向畸变
      \begin{cases}
      k1
      \\
      k2
      \\
      k3
      \end{cases}
      \\
      切向畸变
      \begin{cases}
      p1
      \\
      p2
      \end{cases}
      $$
   
   3. 
   
   ![相机畸变](..\Picture\相机畸变.png)
   
6. 相机外参Extrinsic parameters

   坐标系的转换需要用到相机外参

   

## 2模型求解

1. **内参Intrinsic和单应性矩阵Homography的关系**

   内参的初始估计有**闭环解**，不需要瞎估

   tips:

   - 解析解Analytical solution（封闭解Closed-form solution）：

     根据**严格的公式推导**，给出任意自变量就可以求出其因变量，也就是问题的解

     解析解是一个封闭形式（closed-form）的函数，因此对于任意变量，我们皆可以将其带入解析函数求得正确的因变量。因此解析解也被称为封闭解（closed-form solution）

     

   - 数值解（Numberical solution）

     是采用某种计算方法，如：有限元法、数值逼近法、插值法得到的解。

     给出解的具体函数形式，从解的表达式中就可以算出任何对应值

   ![内参和单应性矩阵的关系](..\Picture\内参和单应性矩阵的关系.png)

   

2. **闭环求解 closed form solution**

   ![闭环求解B和内参和外参](..\Picture\闭环求解B和内参和外参.png)

   

3. **优化**

   做BA，优化重投影误差

   目标函数和优化变量

   ![img](https://pic2.zhimg.com/80/v2-f0e10d55e1d8d581bb9776e37b3ff1dd_720w.jpg)

   优化变量的初始值：闭环解，畸变系数初始为0

   优化方法：G-N或LM





reference:

https://zhuanlan.zhihu.com/p/87334006

https://www.twblogs.net/a/5b8b4a682b717718832e9793/?lang=zh-cn

https://www.cnblogs.com/vive/p/5006552.html