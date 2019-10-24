---
title: jupyter kernel error
date: 2018-06-05 11:30:31
tags: 机器学习
categories: 机器学习
---
## 问题出现
之前因为Python2中混淆了编码问题，Python2的str默认是ascii编码，和unicode编码冲突，解决方法主要有两种，一种是用sys.setdefaultencoding(utf8)来进行强制转换，
还有一种是用区分了unicode str和byte array的Python3。
所以用Anaconda同时安装了Python2.7和Python3.6，但是jupyter notebook却报错如下：
    File”//anaconda/lib/python2.7/site-packages/jupyter_client/manager.py”, line 190, in _launch_kernel 
    return launch_kernel(kernel_cmd, **kw) 
    File “//anaconda/lib/python2.7/site-packages/jupyter_client/launcher.py”, line 123, in launch_kernel 
    proc = Popen(cmd, **kwargs) 
    File “//anaconda/lib/python2.7/subprocess.py”, line 710, in init 
    errread, errwrite) 
    File “//anaconda/lib/python2.7/subprocess.py”, line 1335, in _execute_child 
    raise child_exception 
    OSError: [Errno 2] No such file or director

## 解决方法
1. 首先使用jupyter kernelspec list查看安装的内核和位置
2. 进入安装内核目录打开kernel.jason文件，查看Python编译器的路径是否正确
3. 如果不正确python -m ipykernel install --user重新安装内核，如果有多个内核，如果你使用conda create -n python2 python=2,为Python2.7设置conda变量,那么在anacoda下使用activate pyhton2切换python环境，重新使用python -m ipykernel install --user安装内核
4. 重启jupyter notebook即可