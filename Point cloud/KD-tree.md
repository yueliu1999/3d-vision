## KD-tree

KD树的建立

1. 计算各维度特征的方差，选择方差最大的为分割特征
2. 选择该特征中的中位数为分割点（根节点）
3. 将数据集中该维度特征小于中位数的节点传递给左儿子，大于中位数传递给右儿子
4. 递归执行1-4，直到所有数据集都在KD-Tree的节点上为止



KD Tree和BST很相似，BST为KD Tree在一维上的特例

KD Tree的算法时间复杂度为
$$
时间复杂度\in [O(Log_2(N)), \ O(N))]
$$


reference：

https://zhuanlan.zhihu.com/p/45346117

https://blog.csdn.net/xbmatrix/article/details/63683614