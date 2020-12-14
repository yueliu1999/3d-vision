# git&Github



## git命令

- 查看隐藏文件，带.的为隐藏文件：

  ls -la

  

- pwd

  显示当前路径

  

- 新建仓库：

  git init

  

- 项目级别/仓库级别：

  git config user.name tom_pro

  git config user.email 123@qq.com

  信息保存位置：./git/config

  

- 系统用户级别

  git config --global user.name tom_glb

  git cinfig --global user.email 123@qq.com

  信息保存位置：~/.gitconfig

  

- 连接文件并打印到标准输出设备上

  cat config 

  

- 切到home路径

  cd ~

  

- 查看git状态（工作区、暂存区）

  git status

  ![1585311886147](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\1585311886147.png)

  

- 启用vim编辑器

  vim good.txt

  

- vim编辑器加行号

  set:nu

  

- vim进入编辑模式

  i

  

- vim写入退出

  :wq

  

- 将文件添加到暂存区

  git add good.txt

  

- 将文件从暂存区中移除

  git rm --cached good.txt



- 从暂存区提交到本地库

  git commit good.txt

  

- 从暂存区提交到本地库，加上注释

  git commit -m "123" good.txt

  

- 查看记录

  git log

  - 空格向下翻页
  - b向上翻页
  - q退出

  git log --pretty=oneline

  git log --oneline

  git reflog

  

- 改变版本

  基于索引值，可以前进或者后退：

  git reset --hard 14d26f2  

  

  只能后退，不能前进：

  退两步：

  git reset --hard 14d26f2^^

  

  退n步

  git reset --hard~3

  

  --soft，仅仅在本地库移动head指针

  --mixed，在本地库移动head指针，重置暂存区

  --hard，在本地库移动head指针，重置暂存区，重置工作区

  

- 删除

  git add aaa.txt

  git commit -m "delete" aaa.txt

  

- 比较文件差异

  和暂存区比较

  git diff apple.txt

  

  和本地库比较

  git diff head apple.txt

  

  和历史版本比较

  git diff head^ apple.txt

  

  比较所有文件

  git diff

  

- 分支管理

  - 创建分支

    git branch hot_fix

  - 查看分支

    git branch -v

  - 切换分支

    git checkout hot_fix

  - 合并分支

    1. 切换到接受修改的分支

       git checkout master

    2. 指定另一个分支的名字

       git merge hot_fix

    3. 解决冲突

       手动合并，修改完毕后

       git add 文件名

       git commit



## GitHub

- 查看远程库地址

  git remote -v

  

- 添加远程库地址

  git remote add origin https://github.com/YueLiu-coder/China_software_cup_2020.git

  origin 为别名

  

- 从远程库clone文件

  git clone https://github.com/YueLiu-coder/China_software_cup_2020.git

  - 完整的把远程库下载到本地
  - 创建origin远程地址别名
  - 初始化本地库

  

- 抓取

  git fetch origin master

  

- 拉取 

  pull = fetch+merge

  git fetch 远程地址库别名 远程分支名

  git merge 远程地址库别名/分支别名

  git pull origin master



- 推送

  如果不是最新版，则不能推送，先要pull

  pull之后如果进入冲突状态，则进行分支冲突解决

  git push origin master 

  

- 跨团队协作

  1. fork
  2. 本地修改，推送到远程
  3. pull request