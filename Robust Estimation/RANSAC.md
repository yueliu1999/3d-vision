## RANSAC

RANdom SAmple Consensus

随机一致性采样





基本矩阵的求解方法：

1. DLT direct linear transformation

   - 8点法
   - 最小二乘法

   

   要保证有唯一解至少需要8对匹配点

   n=8，若A非奇异(A)，则有唯一解，

   n>=8的时候，可以求解超定方程，得到最小二乘解

   

2. RANSAC RANdom SAmple Concense

   稳健（robust）：对数据噪声的敏感性

   最小二乘对于出现外点，将会影响实际估计效果

   

   RANSAC主要解决样本中的外点问题，最多可以处理**50%的外点情况**

   基本思想：

   通过反复选择数据中的一组随机子集来达成目标，被选取的子集假设为局内点（inliers），并用下述方法进行验证：

   1. 有一个模型适用于假设的inliers，即所有未知的参数都能从假设的inliers中计算得出
   2. 用1中得到的模型去测试所有的其他数据，如果某个点适用于估计的模型，认为它是inliers
   3. 如果有足够多的点被归类为假设的局内点，那么估计的模型就足够合理
   4. 用所有假设的局内点去重新估计模型
   5. 最后通过估计局内点与模型的错误率来评估模型




这个过程被重复执行固定的次数，每次产生的模型要么因为inliers太少而被舍弃，要么有比他更好的模型

![img](https://pic2.zhimg.com/80/v2-d9e4b96fd378243b21c77b39904ef6c5_720w.jpg)



步骤总结：

- 对于N个样本点数
- K是求解模型所需要的最少的点的个数，例如
  - 直线，仅仅需要两个点就可以计算直线的方程
  - 平面，仅仅需要三个点就可以计算平面的方程
  - 三角化过程，仅仅需要两个view就可以计算平面的方程

1. 随机采样K个点
2. 对该K个点拟合模型model
3. 计算其他点到model的距离，若小于阈值d，则当作inlier，统计inliers的个数
4. 重复M次，选择inliers最多的模型，获取到所有的inliers
5. 利用所有inliers重新进行模型的估计（可选），可以使用最小二乘法



例子，RANSAC直线拟合：

1. 随机选取K=2个点

   ![img](https://pic2.zhimg.com/80/v2-67e966c92f04f232010255dc5cd1b92d_720w.jpg)

2. 拟合一条直线

   ![img](https://pic1.zhimg.com/80/v2-3693478f142577031cfc29b9d61e58c8_720w.jpg)

3. 统计内点的个数

   ![img](https://pic3.zhimg.com/80/v2-bd7445a60766817022f8506274f2eeba_720w.jpg)

   

4. 重复上述过程M次，找到内点数最大的模型

   ![img](https://pic2.zhimg.com/80/v2-fcd467425195baccd67f7d8ec6101c2d_720w.jpg)

5. 利用所有的内点重新估计直线

   ![img](https://pic1.zhimg.com/80/v2-7225d7e8e5dd5d6ea19aa560c866dd9c_720w.jpg)





关于重复次数M的选取：

设

- N为样本点数量

- K求解模型需要最少的点的个数
- P表示内点的概率



P<sup>k</sup>为k个点都是内点的概率

1-p<sup>k</sup>为k个点中至少有一个外点（采样失败）的概率

(1-p<sup>k</sup>)<sup>M</sup>为M次采样都失败的概率

z = 1-(1-p<sup>k</sup>)<sup>M</sup>为M次采样至少有一次成功的概率

所以

M = log(1-z)/log(1-p<sup>k</sup>)   利用了换底公式



tips:

- Markdown是一个兼容嵌套HTML的语法，在HTML中上标和下标的语法是\<sub>\</sub>和\<sup>\</sup>



















reference: 

https://zhuanlan.zhihu.com/p/45532306

https://www.jianshu.com/p/13b3366f0260