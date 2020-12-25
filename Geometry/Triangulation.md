# Triangulation

三角化：由像点计算物点的过程称为三角化，在摄影测量领域称为前方交会

![img](https://pic3.zhimg.com/80/v2-bc36bd3c7fdf318ada42c624d732499e_720w.jpg)



对于空间中有>=2个view的可以观察到的点，我们可以采用三角化来获取到他的深度值Z







### 数学推导

若已知：

成像平面上的坐标x和相机外参Twc，则可以利用公式lambda x = TwcX来联立方程，其中可以利用共线向量的叉乘为0来计算，一个相机和一个点可以提供两个方程，X有三个未知量，所以需要利用最少两个视角来计算。可以利用SVD来求出最小二乘解。



若已知

像素坐标上的坐标x和相机内参K和外参Twc，则可以利用公式lambda x = KTwcX来联立方程，其中可以利用共线向量的叉乘为0来计算，一个相机和一个点可以提供两个方程，X有三个未知量，所以需要利用最少两个视角来计算。可以利用SVD来求出最小二乘解。

![Triangulation_1](..\picture\Triangulation_1.png)

![Triangulation_2](..\picture\Triangulation_2.png)

Tips：共线向量的叉乘为0，这是比较常用的方法



### 多种三角化

三类方法：

1. midpoint methods

   minimize the sum of squared distances to each ray

2. linear least squares methods

   minimize the algebraic errors，DLT？direct linear transformation

3. optimal methods

   a cost function based on either image reprojection errors or angular reprojection errors

   L1 norm: sum of magnitude

   L2 norm: sum of squares

   L∞ norm: maximum

   of reprojection errors

   





- 2个views的三角化

  openmvg中四种方法：

  - TriangulateDLT

    direct linear transform

    直接按照数学推导的方法进行方程的构建以及SVD求解

    code from OpenMVG

    ```c++
        void TriangulateDLT
        (
            const Mat34& P1,
            const Vec3& x1,
            const Mat34& P2,
            const Vec3& x2,
            Vec4* X_homogeneous
        )
        {
            // Solve:
            // [cross(x0,P0) X = 0]
            // [cross(x1,P1) X = 0]
            Mat4 design;
            design.row(0) = x1[0] * P1.row(2) - x1[2] * P1.row(0);
            design.row(1) = x1[1] * P1.row(2) - x1[2] * P1.row(1);
            design.row(2) = x2[0] * P2.row(2) - x2[2] * P2.row(0);
            design.row(3) = x2[1] * P2.row(2) - x2[2] * P2.row(1);
    
            const Eigen::JacobiSVD<Mat4> svd(design, Eigen::ComputeFullV);
            (*X_homogeneous) = svd.matrixV().col(3);
        }
    ```


  

  

  - TriangulateL1Angular

    角度重投影误差的L1范数

    angular reprojection error

    @ref S.H. Lee, J. Civera - Closed-Form Optimal Triangulation Based on Angular Errors - ICCV 2019 - https://arxiv.org/pdf/1903.09115.pdf

    

  - TriangulateLInfinityAngular

    角度重投影误差的L无穷范数

    @ref S.H. Lee, J. Civera - Closed-Form Optimal Triangulation Based on Angular Errors - ICCV 2019 - https://arxiv.org/pdf/1903.09115.pdf

  

  pipeline：

  INPUT：进行标定，找到内参矩阵K，相机的相对位姿R、t；找到一对匹配点u0和u1从两个视角c0和c1

  OUTPUT：3d点X在frame C1的坐标系下

  1. f0 = K-1u0，f1 = K-1u1

     m0 = Rf0

     m1 = f1

  2.   

     - 对于L1约束的三角化

       minimize： theta1 + theta2

       

     - 对于L2约束的三角化

       minimize： sine(theta1)^2 + sine(theta2)^2

       

     - 对于L∞约束的三角化

       minimize：max(theta1, theta2)

       

  3. Rf0' = m0'

     f1' = m1'

     

  4. Check cheirality

  5. Check angular reprojection errors

     - theta0 =  <Rf0，Rf0' and theta1 = <f1,f1'

     - discard the point and terminate if 

       max(theta0,theta2)>阈值 for some small 阈值

  6. Check parallax

     - belta = <(Rf0',f1')
     - discard the point and terminate if 

  7. Compute and return X1'

  手写部分：

  ![angular_reprojection_1](D:\3d-vision\picture\angular_reprojection_1.png)

  

  ![angular_reprojection_2](D:\3d-vision\picture\angular_reprojection_2.png)

  

  

  - TriangulateIDWMidpoint

    @ref S.H. Lee, J. Civera - Triangulation: Why Optimize? - BMVC 2019 - https://arxiv.org/pdf/1907.11917.pdf

    

    
    
    
    
    
    
    
    
    




- N个views的三角化

  





Ransac三角化











reference:

https://zhuanlan.zhihu.com/p/55530787









