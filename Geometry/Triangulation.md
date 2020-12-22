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

- 2个views的三角化

  openmvg中四种方法：

  - TriangulateDLT
  - TriangulateL1Angular
  - TriangulateLInfinityAngular
  - TriangulateIDWMidpoint
  - 

  

- N个views的三角化





Ransac三角化











reference:

https://zhuanlan.zhihu.com/p/55530787









