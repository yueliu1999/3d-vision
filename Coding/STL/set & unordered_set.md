## set & unordered_set

## set

stl中set是有序容器，可以通过传入自定义的比较器对象的方式，设定想要使用的比较方法



使用迭代器遍历set的时候，遍历的顺序就说比较器定义的顺序



```c++
set<int> s;
// 插入的时候按照从大到小的顺序插入
for (int i = 10; i > 0; i--)
{
    s.insert(i);
}
set<int>::iterator it;
// 遍历的时候的输出是从小到大
for (it = s.begin(); it != s.end(); ++it)
{
    cout << *it ;
}
```





可以通过传入比较器函数对象的形式，更改set排序方式：

```c++
// 从大到小排序的比较器函数对象
struct Compartor
{
    bool operator()(const int lhs,const int rhs) const
    {
        return rhs < lhs;
    }
};
// 声明使用自定义比较器的set
set<int,Compartor> s;
// 按照从小到大的顺序插入
for (int i = 0; i < 10; i++)
{
    s.insert(i);
}
set<int>::iterator it;
// 输出的顺序的作用是从大到小
for (it = s.begin(); it != s.end(); ++it)
{
    cout << *it ;
}
```

## unordered_set

这种set是不会进行排序的，顺序和插入顺序一致

