## Filter

在图像处理或者计算机视觉的应用中，一般会需要一个预处理的过程。

图像本身就是一个二维的信号，其中像素点灰度值的高低代表信号的强弱：

- 高频：图像中灰度变化剧烈的点，一般是图像中物体的轮廓或者噪声
- 低频：图像中平坦的，灰度变化不大的点，图像中大部分的区域



设计滤波器（Filter）:

- 高通滤波器：可以检测图像中尖锐、变化明显的地方

  - 基于Canny边缘滤波

  - 基于Sobel边缘滤波

    

- 低通滤波器：可以使图像变得光滑，滤除图像中的噪声

  - 线性的均值滤波
  - 中值滤波器
  - 高斯滤波器
  - 非线性双边滤波



低通滤波器和高通滤波器是互相矛盾的，但是很多时候在做边缘检测之前我们需要用低通滤波来进行降噪，**需要调节参数在保证高频的边缘不丢失的前提下尽可能多的去除图片的噪点**



扫描的方式：卷积

核：固定大小的数值矩阵，该数组带有一个锚点anchor，位于矩阵中央。如下图所示：

![img](https://img-blog.csdn.net/20160319162457966?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

核可以是opencv已经定义好的均值滤波器核和高斯滤波器核，也可以自定义核



**卷积**的计算方式：

1. 将核的anchor放在特定位置的像素上，其余位置与该像素邻域的各像素重合
2. 核内各值与相应的像素相乘，并将乘积相加
3. 将计算所得到的结果放在anchor对应的像素上
4. 对图像所有的像素重复上述过程



公式：

![img](https://img-blog.csdn.net/20160319162955894)



![img](https://img-blog.csdn.net/20160319001312814?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



计算图像大小的公式：

(src-kernel_size+2*padding)/stride+1





1. 均值滤波器

   **平滑线性空间滤波器**的输出是包含在滤波器模板领域内的像素的简单平均值，也就是均值滤波器。

   

   均值滤波器是**低通滤波器**，把领域内的平均值赋值给中心元素

   

   作用：

   - 降低噪声
   - 去除图像中的不相关细节，不相关是指与**滤波器模板相比较小**的像素区域
   - 模糊图片以便得到感兴趣物体的粗略描述，因此那些较小的物体的灰度会与背景混合在一起，较大的物体则变得像斑点而易于检测

   

   OpenCV函数: 

   ```c++
   void blur(InputArray src, OutputArray dst, Size ksize, Point anchor = Point(-1,-1), int borderType = BORDER_DEAFAULT);
   /*
   参数说明:
   参数1: 输入图像
   参数2: 输出图像
   参数3: 内核大小
   参数4: 锚点（被平滑的点），默认值为Point(-1,-1)表示为中心点
   参数5: 边界模式，默认值为BORDER_DEFAULT
   */
   ```

   核：

   ![img](https://img-blog.csdn.net/20160319003603677)

   

   

   

   

   

2. 中值滤波器

   是一种**非线性滤波器**，常用于消除图像的**椒盐噪声**

   与**低通滤波**不同，中值滤波有利于保留边缘的尖锐度，但它会洗去均匀介质区域中的纹理。

   

   椒盐噪声是由**图像传感器**，传输信道，解码处理等产生的黑白相间的亮暗点噪声。

   

   椒盐噪声是指两种噪声

   - 盐噪声

     salt noise = 255，白色，高灰度噪声

   - 胡椒噪声

     pepper noise = 0，黑色，低灰度噪声

     

   一般两种噪声同时出现的时候，

   - 对于灰度图，会有黑白杂点
   - 对于三通道图像，则表现在单个像素BGR三个通道随机出现的255或0

   tips:

   椒盐噪声也称为脉冲噪声，是图像中经常见到的一种噪声，是随机出现的白点或黑点。成因可能是影像讯号受到突入其来的强烈干扰而产生、类比数位转换器或位元传输错误

   

   用中值代替中心

   

   OpenCV函数：

   ```c++
   void medianBlur(InputArray src,OutputArray dst,int ksize);
   /*
   参数说明:
   参数1:输入图像
   残数2:输出图像
   参数3:内核大小
   */
   ```

3. 高斯滤波器

   用于平滑图像，或者说图像模糊处理，高斯滤波是低通的，变化频率低的会通过

   基本思想：

   图像上每一个像素点的值，都由其本身核邻域内的其他像素点的值经过加权平均后得到。

   高斯滤波的过程是图像与**高斯正太分布**做卷积操作

   ![[公式]](https://www.zhihu.com/equation?tex=G%28x%2Cy%29%3D%5Cfrac%7B1%7D%7B2%5Cpi%5Csigma%7D%2Ae%5E%7B-%5Cfrac%7Bx%5E%7B2%7D%2By%5E%7B2%7D%7D%7B2%2A%5Csigma%5E%7B2%7D%7D%7D)

   

   权值的分布是以中间高，四周低来分布的。并且距离中心越远，其对中心点的影响越小，权值也就越小

   - 核大小固定，sigma越大，权值分布越平缓，因此对各个点的值对输出值影响越大，最终结果造成图像越模糊
   - 核大小固定，sigma越小，权值分布越突起，因此对各个点的值对输出值影响越小，最终结果影响越小，如果中心点权值为1，其他为0，则没有变化
   - sigma固定，核越大，图像越模糊
   - sigma固定，核越小，图像变化越小

   

   OpenCV函数：

   ```c++
   void GaussianBlur(InputArray src,OutputArray dst,Size ksize,double sigmaX,double sigmaY = 0,int borderType = BODER_DEAFAULT);
   /*
   参数1:输入图像
   参数2:输出图像
   参数3:高斯内核的大小。其中width和height可以不同，但是他们必须为正数和奇数，或者他们都可以是0，他们是由sigma计算而来的
   参数4:表示高斯核函数在X方向的标准偏差
   参数5:表示高斯核函数在Y方向上的标准偏差，若sigmaY为0，则设为sigmaX，如果都为0，则由ksize.width和ksize.height计算出来
   参数6:边界模式
   */
   ```

   

4. 双边滤波器

   双边滤波（Bilateral Filter）是非线性滤波的一种，这是一种结合：

   - 图像的空间邻近度（**临近信息**）
   - 像素值相似度（**颜色相似信息**）

   的处理方法

   在滤除噪声、平滑图像的同时，又做到了边缘保存

   **采用了两个高斯滤波的结合**

   - 一个负责**空间邻近度**的权值，常用的高斯滤波器
   - 一个负责计算像素值相似度的权值

   将两者优化的权值进行乘积，再与图像做卷积运算，从而**保边而去噪**

   ![bilateral filter](..\Picture\bilateral filter.png)

   OpenCV函数

   ```c++
   void bilateralFilter(InputArrary src, OutputArray dst,int d, double sigmaColor, double simgaSpace, int borderType = BORDER_DEFAULT);
   /*
   参数说明:
   参数1:输入图像
   参数2:输出图像
   参数3:表示在过滤过程中每个像素邻域的直径范围。如果这个值是非正数，则函数会从第五个参数sigmaSpace计算该值
   参数4:颜色空间过滤器的sigma值，这个参数的值月大，表明该像素邻域内有越宽广的颜色会被混合到一起，产生较大的半相等颜色区域
   参数5:坐标空间过滤器的sigma值
   参数6:边界模式
   */
   ```

   

5. 引导滤波(Guidance Filter)

6. 迭代引导滤波(Rolling Guidance Filter)

7. 加权最小二乘滤波



总结：

- 均值滤波、高斯滤波都是对整体图像进行模糊，不会保持边缘特性
- 中值滤波、边缘滤波、引导滤波、迭代引导滤波、加权最小二乘滤波都有保持边缘的特性







reference:

https://zhuanlan.zhihu.com/p/257298948

https://baike.baidu.com/item/%E6%A4%92%E7%9B%90%E5%99%AA%E5%A3%B0

https://blog.csdn.net/weixin_38570251/article/details/82054106