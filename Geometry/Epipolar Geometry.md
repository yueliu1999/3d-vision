# Epipolar Geometry

对极几何！=双目测距

- 对极几何要求Zc 与 Zc' 相交，

- 两个相机离的太近，或物体太远，会导致夹角太小，损失精度



- 双目测距要求Zc 与 Zc'平行

- 双目距离太近，或物体太远会，精度下降

  Zc = fx * tx/d

  tx为相机之间的距离；d为像素之间的视差

  如果物体距离太远，则视差很小，对于错误匹配的误差就很大





本质矩阵和基础矩阵

- 本质矩阵（Essential matrix）

  E = t x R = [t]x  R

- 基础矩阵（fundamental matrix）

  F = (K1^-1)^T  E  (K2^-1)



手写部分、推导：


![essential_matrix](..\picture\essential_matrix.png)



![fundamental_matrix](..\picture\fundamental_matrix.png)





reference：

https://zhuanlan.zhihu.com/p/141799551

