## Eigen初始化

初始化矩阵的高级方法

### 逗号初始化

全是数字

```c++
Matrix3f m;
m<<
    1,2,3,
	4,5,6,
	7,8,9;
```

由行向量组合

```c++
RowVectorXd vec1(3);
vec1<<1,2,3;
RowVectorXd vec2(4);
vec2<<1,4,9,16;
RowVectorXd joined(7);
joined<<vec1,vec2;
```



### 特殊矩阵和数组

```c++
std::cout << "A fixed-size array:\n";
Array33f a1 = Array33f::Zero();
std::cout << a1 << "\n\n";
std::cout << "A one-dimensional dynamic-size array:\n";
ArrayXf a2 = ArrayXf::Zero(3);
std::cout << a2 << "\n\n";
std::cout << "A two-dimensional dynamic-size array:\n";
ArrayXXf a3 = ArrayXXf::Zero(3, 4);
std::cout << a3 << "\n";
```

```c++
ArrayXXf table(10, 4);
table.col(0) = ArrayXf::LinSpaced(10, 0, 90);
table.col(1) = M_PI / 180 * table.col(0);
table.col(2) = table.col(1).sin();
table.col(3) = table.col(1).cos();
std::cout << "  Degrees   Radians      Sine    Cosine\n";
std::cout << table << std::endl;
```

```c++
const int size = 6;
MatrixXd mat1(size, size);
mat1.topLeftCorner(size/2, size/2)     = MatrixXd::Zero(size/2, size/2);
mat1.topRightCorner(size/2, size/2)    = MatrixXd::Identity(size/2, size/2);
mat1.bottomLeftCorner(size/2, size/2)  = MatrixXd::Identity(size/2, size/2);
mat1.bottomRightCorner(size/2, size/2) = MatrixXd::Zero(size/2, size/2);
std::cout << mat1 << std::endl << std::endl;
MatrixXd mat2(size, size);
mat2.topLeftCorner(size/2, size/2).setZero();
mat2.topRightCorner(size/2, size/2).setIdentity();
mat2.bottomLeftCorner(size/2, size/2).setIdentity();
mat2.bottomRightCorner(size/2, size/2).setZero();
std::cout << mat2 << std::endl << std::endl;
MatrixXd mat3(size, size);
mat3 << MatrixXd::Zero(size/2, size/2), MatrixXd::Identity(size/2, size/2),
        MatrixXd::Identity(size/2, size/2), MatrixXd::Zero(size/2, size/2);
std::cout << mat3 << std::endl;
```





### 临时对象的用法

```c++
#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;
int main()
{
  MatrixXd m = MatrixXd::Random(3,3);
  m = (m + MatrixXd::Constant(3,3,1.2)) * 50;
  cout << "m =" << endl << m << endl;
  VectorXd v(3);
  v << 1, 2, 3;
  cout << "m * v =" << endl << m * v << endl;
}
```



reference:

https://blog.csdn.net/u012936940/article/details/79830267