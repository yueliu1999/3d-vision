## Binocular stereo vision

双目立体视觉是机器视觉的一种重要形式，基于视差原理并利用成像设备从不同的位置获取被测物体的两幅图片，通过计算图像对应点间的位置偏差，来获取物体三维信息的方法



计算三维点云的空间坐标：
$$
\frac{X}{D} = \frac{x-x_0}{f_x}
\\
\frac{Y}{D} = \frac{y-y_0}{f_y}
\\
Z = D
$$


深度D非常重要

计算深度D的公式：
$$
D = \frac{Bf}{d}
\\
B是基线的长度，f是焦距，d是视差
$$

- 分母是像素为单位，和算法相关
- 分子是硬件参数，基线和焦距，和硬件相关





### 1 算法因素

$$
D = \frac{Bf}{d}
$$

硬件因素B和f恒定了

假设视差偏差为$\Delta d$
$$
D + \Delta D= \frac{Bf}{d+\Delta d}
$$

$$
\Delta D = Bf(\frac{1}{d}-\frac{1}{1+\Delta d})
$$

$\Delta d$越小，$\Delta D$越小

**所以是视差偏差越小，深度偏差越小**



### 2 硬件因素

$\Delta D = Bf(\frac{1}{d}-\frac{1}{1+\Delta d})$

其中$\Delta D = D - \frac{1}{\frac{1}{D}+\frac{\Delta d}{Bf}}$

**基线越大，焦距越长，深度精度越高**



像素大小越小，同样物理尺寸焦距有更长的像素尺寸焦距，深度精度就越高



[教你提高双目立体视觉系统的精度](https://blog.csdn.net/rs_lys/article/details/107102968)