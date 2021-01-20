## Blue noise sampling

图像采样中的蓝噪声方法



蓝噪声是指**均匀分布的随机无偏噪声**

在图像领域有广泛应用



生成蓝噪声的方法：

DART方法



**图像采样：**

在图像内选取若干个点作为采样点，用这些采样点的情况（**色彩、透明度**）描述点对应的周围一个区域的整体色彩情况。



**蓝噪声采样：**

生成n维空间内符合蓝噪声标准的均匀分布的一系列点，对图像（n维空间数据）进行采样，以使得采样尽可能的均匀。





**为什么不使用均匀采样/随机（均匀分布）采样的方法？**

随机采样6667个点：

![img](https://bost.ocks.org/mike/algorithms/uniform-random-voronoi.jpg)



DART算法

![img](https://bost.ocks.org/mike/algorithms/best-candidate-voronoi.jpg)



Bridson方法

![img](https://bost.ocks.org/mike/algorithms/poisson-disc-voronoi.jpg)







reference：

https://zodiac911.github.io/blog/blue-noise.html

https://www.labri.fr/perso/nrougier/from-python-to-numpy/





