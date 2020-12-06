# 奇异值分解（SVD）

Singular Value Decomposition

用处：

- 降维算法中的特征分解

- 用于推荐系统

- 自然语言处理领域





## 1、回顾特征值和特征向量

A x = lambda x

A：n*n的矩阵

x：n维列向量

则lambda为矩阵A的一个特征值，而x为矩阵A的特征值lambda所对应的特征向量



求出**特征值lambda**和**特征向量x**的好处：

可以将矩阵A进行特征分解表示

A = WΣW^-1

其中W是n个特征向量所张成的n*n维矩阵，而Σ是n个特征值为主对角线的n\*n维的矩阵，一般W的这n个特征向量标准化，即满足||Wi||=1，或Wi^T\*Wi = 1，此时W的n个特征向量为标准正交基，满足W^TW = I，即W^T = W^-1，就算是W为酉矩阵。

A = WΣW^T

所有要进行特征分解，矩阵A必须是方阵！



如果不是方阵，行数！=列数时候，使用SVD！



## 2、SVD的定义

SVD也是对矩阵进行分解，SVD不要求为方阵，假设A为m*n的矩阵，那么定义矩阵A的SVD为：

A = UΣV^T

U：m*m矩阵

Σ：m*n矩阵， 除了主对角线上的元素以外全为0，主对角线上每个元素为奇异值

V：n*n矩阵



U和V都是酉矩阵

U^TU = I

V^TV = I

![img](https://pic4.zhimg.com/80/v2-5ee98f8f3426b845bc1c5038ecd29593_720w.jpg)



求解U、Σ、V：



求U：

A^T * A为n*n的矩阵，方阵

A^TA * Vi = lambdai * Vi

得到矩阵A^T*A的n个特征值和n个特征向量，A^T * A的所有特征值向量张成一个n*n的矩阵V，则就SVD中的V。一般V中的每个特征向量为右奇异向量



求V：

A * A^T为m*m的矩阵，方阵

AA^T * Ui = lambdai * Ui

得到矩阵AA^T的m个特征值和m个特征向量，AA^T的所有特征值向量张成一个m*m的矩阵U，则就SVD中的U。一般U中的每个特征向量为左奇异向量

 

求Σ：

Σ需要求出每个奇异值

A = UΣV^T

AV= UΣ

Σ = U^TAV



证明：

A^TA = VΣU^TUΣV^T = V Σ^2 V^T，所有A^TA的W为V，特征值为奇异值的平方

AA^T = U^TΣVV^TΣU = U Σ^2 U^T，所有AA^T的W为U，特征值为奇异值的平方

所以求奇异值可以直接用特征值的根号



## 3、SVD计算





















