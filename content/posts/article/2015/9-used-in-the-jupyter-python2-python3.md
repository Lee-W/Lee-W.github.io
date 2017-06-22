Title: 在jupyter中同時使用Python2 Python3
Date: 2015-11-02 15:12
Category: Python
Tags: Jupyter
Slug: used-in-the-jupyter-python2-python3
Authors: Lee-W
Summary: 


先安裝Python2和Python3的ipython notebook
```shell
pip2 install ipython notebook
pip3 install ipython notebook
```

分別用各自的ipython執行下面的指令
```shell
ipython2 kernelspec install-self
ipython3 kernelspec install-self
```
<!--more-->

就能在ipython notebook裡面同時使用兩種版本的Python了
![1_jupyter](http://i.imgur.com/IxopQfG.png)
Python2上面是另一個也被jupyter notebook支援的語言julia
最近才剛開始碰，有機會再來分享julia的心得

# Reference
[Using both Python 2.x and Python 3.x in IPython Notebook](http://stackoverflow.com/questions/30492623/using-both-python-2-x-and-python-3-x-in-ipython-notebook)