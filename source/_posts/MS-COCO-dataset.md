---
title: MS-COCO dataset
date: 2018-09-17 22:28:43
tags: [深度学习,数据集]
categories: 深度学习
---
最近在复现deeplab-v3时想在MS-COCO数据集上pretrain，所以鼓捣了很久的数据集，做个记录。

# 数据集简介及下载

MS-COCO是微软持续更新的一个可以用来图像recognition+segmentation+captioning的数据集，
其中训练集有118287张图片和测试集有5000张图片，其官方说明地址：http://mscoco.org/
<!-- more -->
官网被墙了，所以直接从下载链接下载，速度还可以。linux下wget很方便，建议用断点续传命令 wget -c http
[训练集](http://images.cocodataset.org/zips/train2017.zip)
[训练集注释](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

[验证集](http://images.cocodataset.org/zips/val2017.zip)
[验证集注释](http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip)

[测试集](http://images.cocodataset.org/zips/test2017.zip) 
[测试集注释](http://images.cocodataset.org/annotations/image_info_test2017.zip)

由于云服务器是内部网的(T_T)，所以先ssh到本地的服务器，然后通过堡垒机作为中转终于传到云服务器上了。

# cocoAPI

coco数据集的注释是以json格式存储的，coco配置了数据读取的API，下载链接：https://github.com/nightrome/cocostuffapi

下载之后cd到PythonAPI路径下执行命令make就行

我在用的时候碰到了gcc错误

```
python setup.py build_ext --inplace
running build_ext
building 'pycocotools._mask' extension
gcc -pthread -fno-strict-aliasing -g -O2 -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/home/rizqi-okta/miniconda2/lib/python2.7/site-packages/numpy/core/include -I../common -I/home/rizqi-okta/miniconda2/include/python2.7 -c pycocotools/_mask.c -o build/temp.linux-x86_64-2.7/pycocotools/_mask.o -Wno-cpp -Wno-unused-function -std=c99
pycocotools/_mask.c:547:21: fatal error: maskApi.h: No such file or directory
compilation terminated.
error: command 'gcc' failed with exit status 1
Makefile:3: recipe for target 'all' failed
make: *** [all] Error 1
```

也重装了Cython，还是不行，最后发现是自己挪动了PythonAPI文件夹的位置，从而找不到_maskApi.c等配置文件，重新调整路径后终于搞定。
如果有遇到其他问题，可以参考这个issue：https://github.com/cocodataset/cocoapi/issues/141

# 用cocoAPI来生成segmentation的label

因为我的环境是python3，所以在以下地方有修改：
1. 在PythonAPI/cocostuff/cocoSegmentationToPngDemo.py中line.68 修改xrange为range
2. 在PythonAPI/cocostuff/cocoSegmentationToPngDemo.py中注释掉line.62-64，这是示例限制生成label个数
   再注释掉line.76-88
3. 设置好annPath为自己的stuff_train/val2017.json路径，运行即可

## 参考文献

- [MS-COCO数据集来做semantic segmentation](https://blog.csdn.net/qq_33000225/article/details/78985635)
- [Microsoft COCO Captions: Data Collection and Evaluation Server](https://arxiv.org/abs/1504.00325)
   
   