# tangram_solve
this method is not universially compatible for all inputs, but most inputs can be solved.
此方法并不能求解所有情况的输入，但大部分输入都可以被快速求解。
## 文件架构
- tangram_solve文件夹中是方案二的python代码和ui界面的设计代码；方案二是ui可执行程序所使用的方案
- tangram_solve_contour文件夹中是方案一的python代码，可运行game_class.py查看效果
- tangram_app文件夹中是打包的可执行程序，运行“七巧板解答.exe”。该文件夹内的tangrams文件夹是必做任务一的图形文件，不能被改动，否则exe文件无法运行。
## 代码架构
### game_class.py
搜索算法核心代码，包括以下自定义类：

- Node_open_list：存储在open表中的搜索节点，保存搜索节点的父节点序号，全部的基本三角形信息，各个板块占据的基本三角形序号等。
- Solver：搜索所用的类，保存初始图像的Graph对象，open表，求解轨迹等，有solve_dfs2深度优先搜索方法及相关的一系列生成后继节点所用函数。
### pic_process.py
图像处理部分的核心代码，包括以下自定义类：

- Graph:存储输入图像，以及输入图像的图形节点，基本小三角形等。定义有获取图形节点、基本小三角形的相关方法。
- Pic feature: 图形节点类，存储图形节点的坐标，以及8个邻居图形节点，相关基本三角形，其中相关基本三角形包括以该点为90度角顶点的基本三角形和以该点为45度角顶点的基本三角形。
- Small tri：基本三角形类，存储图形中的基本三角形的三个顶点序号。

除了这些类之外还有距离、面积测量，图形可视化等函数。

### ui.py
UI界面的顶层Widget
### ui_basic.py
“求解七巧板”标签页的ui设计
### ui_userDefine.py
“用户自定义”标签页的ui设计
### ui_general.py
“任意图形拼接”标签页的ui设计

## 编译运行环境
python 3.6, pyqt 5.13.1, opencv-python 3.4.3.18, numpy 1.15.2, Pyinstaller 3.5
