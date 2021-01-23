## numpy 随机数

numpy.random库

1. rand基本用法

   产生[0, 1)均匀分布的随机浮点数，其中d为数组的维度

   ```python
   
   #产生形状为(2,)的数组，也就是相当于有2个元素的一维数组。
   temp=np.random.rand(2)
   print(temp)  #[0.70284298 0.40041697]
   print(type(temp)) # 查看数据类型，<class 'numpy.ndarray'>
   print(temp[0])  #查看第一个数
   print(type(temp[0])) #查看具体元素的数据类型，<class 'numpy.float64'>
    
   '''
   结果如下：
   [0.70284298 0.40041697]
   <class 'numpy.ndarray'>
   0.7028429826756175
   <class 'numpy.float64'>
   '''
   ```

2. randn基本用法

   返回正态分布，均值0，方差为1，d为维度

   ```python
   arr1=np.random.randn(2,4)  #二行四列，或者说一维大小为2，二维大小为4
   #均值为0，方差为1
   print(arr1)
   print(type(arr1)) #<class 'numpy.ndarray'>
    
   arr2=np.random.rand()
   print(arr2) #0.37338593251088137
   print(type(arr2))  #<class 'float'>
    
    
   '''
   结果如下：
   [[ 0.56538481  0.41791992  0.73515441  1.73895318]
    [ 2.27590795 -1.17933538 -1.02008043  0.15744222]]
   <class 'numpy.ndarray'>
   0.37338593251088137
   <class 'float'>
   '''
   ```

3. 指定期望和方差的正态分布

   ```python
   #Two-by-four array of samples from N(3, 6.25):
   arr3=2.5 * np.random.randn(2,4)+3  #2.5是标准差，3是期望
   print(arr3)
    
    
   """
   结果如下：
   [[ 2.58150052  6.20108311  1.58737197  9.64447208]
    [ 2.68126136  0.63854145 -1.34499681  1.68725191]]
   """
   ```

4. random基本用法和rand的辨别

   ```python
   import numpy as np
    
   x1=np.random.random()
   print(x1)  #0.14775128911185142
   print(type(x1))  #<class 'float'>
    
   x2=np.random.random((3,3))
   print(x2)
   '''
   [[0.07151945 0.00156449 0.66673237]
    [0.89764384 0.68630955 0.21589147]
    [0.50561697 0.27617754 0.5553978 ]]
   '''
   print(type(x2))  #<class 'numpy.ndarray'>
   print(x2[1,1])  #0.68630955
   ```

5. randint基本用法

   用于生成指定范围内的整数

   ```python
   #产生一个[0,10)之间的随机整数
   temp1=np.random.randint(10)
   print(temp1)
   print(type(temp1))  #<class 'int'>
    
   '''
   5
   <class 'int'>
   '''
   ```

6. uniform 基本用法

   均匀分布

   ```python
   #默认产生一个[0,1)之间随机浮点数
   temp=np.random.uniform()
   print(temp) #0.9520851072880187
   ```

7. seed的用法

   如果seed相同，则随机数序列相同

   ```python
   np.random.seed(10)
   temp1=np.random.rand(4)
   print(temp1)
   np.random.seed(10)
   temp2=np.random.rand(4)
   print(temp2)
    
   #这句就不一样的，因为仅作用于最接近的那句随机数产生语句
   temp3=np.random.rand(4)
   print(temp3)
    
   '''
   [0.77132064 0.02075195 0.63364823 0.74880388]
   [0.77132064 0.02075195 0.63364823 0.74880388]
   [0.49850701 0.22479665 0.19806286 0.76053071]
   '''
   ```

   

