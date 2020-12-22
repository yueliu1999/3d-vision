# 三维点云配准（注册）

Point Cloud Registration

算法：迭代最近点算法及各种变种ICP（Iterative Closest Point）算法



1. 问题定义：

   输入两幅点云，source，target

   输出一个T = [R|t]（考虑刚性变换，可以不是刚性变换）

   使得T(source)和target的重合程度尽可能高

   

   分类：

   - 粗配准（Coarse Registration）

     两幅点云之间的变换完全未知的情况下进行较为粗糙的配准，目的是为了给精配准提供较好的初值

   - 精配准（Fine Registration）

     给一个初始的变换，进一步优化得到更精确的变换

     

2. 算法描述

   当T位刚性变换时，点云配准问题可以描述为

   R*，t\* = argmin(1/|source|)Σ||target_i-(R\*source_i+t)||^2

   

   若知道点云的对应关系，则可以用最小二乘来求解R, t参数

   如何知道对应关系？

   如果我们已经知道了一个大概靠谱的R, t，那么我们可以通过贪心的方式来找两幅点云上点的对应关系（直接找**距离最近**的点作为对应点）

   

   

   算法流程：

   1. 点云预处理

   2. 匹配

   3. 加权

   4. 剔除不合理的对应点对的权重

   5. 计算loss

   6. 最小化loss，求解当前最优变换

   7. 回到2迭代，收敛

   

   子问题：

   - 找最近对应点（Find closet point）

     

   - 找最优变换(Find Best Transform)

   

3. 优缺点和一些改进的算法

   ICP优点：

   - 简单，不必对点云进行分割和特征提取
   - 初值较好的情况下，精度和收敛性都不错

   ICP缺点：

   - 找最近对应点的计算开销大
   - 只考虑了点和点距离，缺少对点云结构信息的利用

   ICP改进的算法：

   - Point-to-Plane ICP
   - Plane-to-Plane ICP
   - Gerneralized ICP (GICP): 综合考虑了point-to-point、point-to-plane和plane-to-plane策略，精度和鲁棒性都有所提高
   - Normal Iterative Closest Point(NICP), 考虑法向量和局部曲率，更进一步利用点云的局部信息。

   

   

   

4. Tricks：

   - 点太多先做降采样
   - 找到一些anchor点对，帮助加速收敛
   - 对应用场景引入一些合理的假设，比如限制旋转、平移的范围，变换自由度数量等



<img src="../picture/ICP.png" alt="ICP" style="zoom:40%;" />









## SVD求解ICP

c++，code from 高翔， 视觉SLAM十四讲

```c++
void pose_estimation_3d3d(const vector<Point3f>& pts1,
                          const vector<Point3f>& pts2,
                          Mat& R, Mat& t)
{
    // center of mass
    Point3f p1, p2;
    int N = pts1.size();
    for (int i=0; i<N; i++)
    {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 /= N;
    p2 /= N;

    // subtract COM
    vector<Point3f> q1(N), q2(N);
    for (int i=0; i<N; i++)
    {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i=0; i<N; i++)
    {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x,
                q2[i].y, q2[i].z).transpose();
    }
    cout << "W=" << W << endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    cout << "U=" << U << endl;
    cout << "V=" << V << endl;

    Eigen::Matrix3d R_ = U * (V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // convert to cv::Mat
    R = (Mat_<double>(3, 3) <<
            R_(0, 0), R_(0, 1), R_(0,2),
            R_(1, 0), R_(1, 1), R_(1,2),
            R_(2, 0), R_(2, 1), R_(2,2));
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}
```







reference:

https://zhuanlan.zhihu.com/p/104735380

https://zhuanlan.zhihu.com/p/107218828