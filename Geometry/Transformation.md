# Transformation

这是图形学中的一些基本的变换，在multi view geometry中也被广泛的应用



变换（transform）是这样的操作：

接收实体：points，vectors，colors

以某种方式转化他们

All translation, rotation, scaling, reflection, and shearing matrices are affine.

![img](https://pic3.zhimg.com/80/v2-232678e983630e35a30cff87a2fece92_720w.jpg)



## 基本变换（Basic Transforms）



- 平移（Translation）

  ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BT%7D%28%5Cmathbf%7Bt%7D%29+%3D+%5Cmathbf%7BT%7D%28t_x%2C+t_y%2C+t_z%29+%3D+%5Cleft%28+%5Cbegin%7Bmatrix%7D+1+%26+0+%26+0+%26+t_x+%5C%5C+0+%26+1+%26+0+%26+t_y+%5C%5C+0+%26+0+%26+1+%26+t_z+%5C%5C+0+%26+0+%26+0+%26+1+%5Cend%7Bmatrix%7D+%5Cright%29)

  

  ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BT%7D%5E%7B-1%7D%28%5Cmathbf%7Bt%7D%29+%3D+%5Cmathbf%7BT%7D%28-%5Cmathbf%7Bt%7D%29%3D%5Cleft%28+%5Cbegin%7Bmatrix%7D+1+%26+0+%26+0+%26+-t_x+%5C%5C+0+%26+1+%26+0+%26+-t_y+%5C%5C+0+%26+0+%26+1+%26+-t_z+%5C%5C+0+%26+0+%26+0+%26+1+%5Cend%7Bmatrix%7D+%5Cright%29+)

  

- 旋转（Rotation）

  二维空间：

  ![[公式]](https://www.zhihu.com/equation?tex=+%5Cbegin%7Baligned%7D+%5Cmathbf%7Bu%7D+%26%3D+%5Cleft%28++%5Cbegin%7Bmatrix%7D+r%5Ccos%28%5Ctheta%2B%5Cphi%29%5C%5C+r%5Csin%28%5Ctheta%2B%5Cphi%29+%5Cend%7Bmatrix%7D+%5Cright%29+%3D+%5Cleft%28+%5Cbegin%7Bmatrix%7D+r%28%5Ccos%5Ctheta%5Ccos%5Cphi+-+%5Csin%5Ctheta%5Csin%5Cphi%29%5C%5C+r%28%5Csin%5Ctheta%5Ccos%5Cphi+%2B+%5Ccos%5Ctheta%5Csin%5Cphi%29+%5Cend%7Bmatrix%7D+%5Cright%29%5C%5C+%26%3D+%5Cunderbrace%7B+%5Cleft%28++%5Cbegin%7Bmatrix%7D+%5Ccos%5Cphi+%26+-%5Csin%5Cphi%5C%5C+%5Csin%5Cphi+%26+%5Ccos%5Cphi+%5Cend%7Bmatrix%7D+%5Cright%29+%7D_%7B%5Cmathbf%7BR%7D%28%5Cphi%29%7D+%5Cunderbrace%7B+%5Cleft%28++%5Cbegin%7Bmatrix%7D+r%5Ccos%5Ctheta%5C%5C+r%5Csin%5Ctheta+%5Cend%7Bmatrix%7D+%5Cright%29+%7D_%7B%5Cmathbf%7Bv%7D%7D+%3D+%5Cmathbf%7BR%7D%28%5Cphi%29%5Cmathbf%7Bv%7D+%5Cend%7Baligned%7D%5Ctag%7B4.4%7D)

  三维空间：

  绕x，y，z轴，其值为：

  ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BR%7D_x%28%5Cphi%29+%3D+%5Cleft%28+%5Cbegin%7Bmatrix%7D+1+%26+0+%26+0+%26+0%5C%5C+0+%26+%5Ccos%5Cphi%26+-%5Csin%5Cphi%26+0%5C%5C+0+%26+%5Csin%5Cphi%26+%5Ccos%5Cphi%26+0%5C%5C+0+%26+0+%26+0+%26+1+%5Cend%7Bmatrix%7D+%5Cright%29%5Ctag%7B4.5%7D)

  ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BR%7D_y%28%5Cphi%29%3D%5Cleft%28+%5Cbegin%7Bmatrix%7D+%5Ccos%5Cphi%26+0+%26+%5Csin%5Cphi%26+0%5C%5C+0+%26+1+%26+0+%26+0%5C%5C+-%5Csin%5Cphi%26+0+%26+%5Ccos%5Cphi%26+0%5C%5C+0+%26+0+%26+0+%26+1+%5Cend%7Bmatrix%7D+%5Cright%29%5Ctag%7B4.6%7D)

  ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7BR%7D_z%28%5Ctheta%29+%3D+%5Cleft%28+%5Cbegin%7Bmatrix%7D+%5Ccos%5Cphi%26+-%5Csin%5Cphi%26+0+%26+0%5C%5C+%5Csin%5Cphi%26+%5Ccos%5Cphi%26+0+%26+0%5C%5C+0+%26+0+%26+1+%26+0%5C%5C+0+%26+0+%26+0+%26+1+%5Cend%7Bmatrix%7D+%5Cright%29%5Ctag%7B4.7%7D)

  R矩阵，3*3,他们的迹是一个常量

  不依赖于旋转轴：

  tr(R) = 1+2 cos∠

  

  N阶矩阵的迹是指矩阵对角线上元素的和

  这三种轴变换矩阵结合（即矩阵相乘）可以表示任意旋转矩阵

  所有旋转矩阵都是正交矩阵且行列式为1

  

  

- 缩放（Scaling）

  S(s) = S(sx,sy,sz)

  ![[公式]](https://www.zhihu.com/equation?tex=+%5Cmathbf%7BS%7D%28%5Cmathbf%7Bs%7D%29%3D%5Cleft%28+%5Cbegin%7Bmatrix%7D+s_x+%26+0+%26+0+%26+0%5C%5C+0+%26+s_y+%26+0+%26+0%5C%5C+0+%26+0+%26+s_z+%26+0%5C%5C+0+%26+0+%26+0+%26+1+%5Cend%7Bmatrix%7D+%5Cright%29%5Ctag%7B4.10%7D+)

  缩放矩阵可以用于放大或缩小物体

  若sx = sy = sz，即三个方向上的所放量是一样的，则称为uniform scaling，否则称为nonuniform scaling

  S只对xyz轴的缩放有效



- 错切（Shearing）

  6个基本的错切变换矩阵

  Hxy

  Hxz

  Hyx

  Hyz

  Hzx

  Hzy

  

  第一个下标用于表示哪一个坐标被错切矩阵改变，第二个表示完成错切的坐标

  ![[公式]](https://www.zhihu.com/equation?tex=+%5Cmathbf%7BH%7D%7Bxz%7D%28s%29%3D%5Cleft%28+%5Cbegin%7Bmatrix%7D+1+%26+0+%26+s+%26+0%5C%5C+0+%26+1+%26+0+%26+0%5C%5C+0+%26+0+%26+1+%26+0%5C%5C+0+%26+0+%26+0+%26+1+%5Cend%7Bmatrix%7D+%5Cright%29%5Ctag%7B4.15%7D+)

  

  

- 变换连接（Concatenation of Transforms）

  矩阵不遵守交换定律，所以顺序非常重要

  ![img](https://pic4.zhimg.com/80/v2-c1c5f206700a09a6b76cc8b2640a797b_720w.jpg)

  把所有变换转为一个矩阵的好处是可以提升效率

  一般的变换顺序是先**缩放**，再**旋转**，再**平移**

  

  C = TRS

  实际计算的时候为

  TRSp = (T(R(S(p))))

  

  

- 刚体变换（The Rigid-Body Transform）

  ![[公式]](https://www.zhihu.com/equation?tex=+%5Cmathbf%7BX%7D%3D%5Cmathbf%7BT%7D%28%5Cmathbf%7Bt%7D%29%5Cmathbf%7BR%7D%3D%5Cleft%28+%5Cbegin%7Bmatrix%7D+r_%7B00%7D+%26+r_%7B01%7D+%26+r_%7B02%7D+%26+t_x%5C%5C+r_%7B10%7D+%26+r_%7B11%7D+%26+r_%7B12%7D+%26+t_y%5C%5C+r_%7B20%7D+%26+r_%7B21%7D+%26+r_%7B22%7D+%26+t_z%5C%5C+0+%26+0+%26+0+%26+1+%5Cend%7Bmatrix%7D+%5Cright%29%5Ctag%7B4.17%7D)

  只需要将R矩阵变换转置矩阵，平移分量变换相反数即可

  

  

- 法线变换（normal transform）

- 逆的计算（computation of inverses）











reference: https://zhuanlan.zhihu.com/p/96717729



