---
title: semantic-segmentation Implementation
date: 2018-11-23 15:42:10
tags: [深度学习, semantic-segmentation]
categories: 深度学习
---
实习结束啦，终于有时间来整理下这几个月做的事情，主要是针对语义分割方向的一些模型复现和改进。
<!-- more -->
# Deeplab-v3

原文地址：[DeepLabv3](http://arxiv.org/abs/1706.05587)
代码：[deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab)

## Abstract

DeepLabv3结构采用了空洞卷积级联或不同采样率空洞卷积并行的架构，以解决多尺度下的目标分割问题。
此外，运用了ASPP(Atrous Spatial Pyramid Pooling)模块，它可以获取多个尺度上的卷积特征，从而进一步提升性能。

## introduction

![summary](/img/deeplabv3-1.jpg)
- a. Image Pyramid：将输入图片放缩成不同比例，分别应用到DCNN上，将预测结果融合得到最终输出
- b. Encoder-Decoder：利用Encoder阶段的多尺度特征，运用到Decoder阶段上恢复空间分辨率(代表工作有FCN、SegNet、PSPNet等工作)
- c. Deeper w.Atroous Convolution：在原始模型的顶端增加额外的模块，例如DenseCRF，捕捉像素间长距离信息
- d. Spatial Pyramid Pooling：空间金字塔池化具有不同采样率和多种视野的卷积核，能够以多尺度捕捉对象

## Model Architecture

![deeplabv3架构](/img/deeplabv3-2.jpg)
Deeplab-v3用ResNet来当作主要提取特征网络，在block4的地方采用了ASPP模块(这个选择在论文中有论述，
分别对比了截取到不同的block的准确度，最终选择了block4)，当output_stride=16时，改进后的ASPP包括
一个1x1 convolution和三个3x3 convolutions，其中3x3 convolutions的atrous rates=(6,12,18)，(所有的filter个数为256，并加入batch normalization)当output_stride=8
时，rates将加倍。然后连接所有分支的最终特征，输入到另一个1x1 convolution(所有的filter个数也为256，然后进行batch normalization)，
再进入最终的1x1 convolution，得到logits结果。

## realization

ASPP的模块如下：

``` python
def atrous_spatial_pyramid_pooling(inputs, output_stride, batch_norm_decay, is_training, depth=256):
  """Atrous Spatial Pyramid Pooling.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    is_training: A boolean denoting whether the input is for training.
    depth: The depth of the ResNet unit output.

  Returns:
    The atrous spatial pyramid pooling output.
  """
  with tf.variable_scope("aspp"):
    if output_stride not in [8, 16]:
      raise ValueError('output_stride must be either 8 or 16.')

    atrous_rates = [6, 12, 18]
    if output_stride == 8:
      atrous_rates = [2*rate for rate in atrous_rates]

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
      with arg_scope([layers.batch_norm], is_training=is_training):
        inputs_size = tf.shape(inputs)[1:3]
        # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
        # the rates are doubled when output stride = 8.
        conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
        conv_3x3_1 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
        conv_3x3_2 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
        conv_3x3_3 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

        # (b) the image-level features
        with tf.variable_scope("image_level_features"):
          # global average pooling
          image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
          # 1x1 convolution with 256 filters( and batch normalization)
          image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
          # bilinearly upsample features
          image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
        net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

        return net
```

## pretrained on MS-COCO

paper上提到了pretrained on MS-COCO的结果，但是我看了下github上好像没有类似实现，所以自己实现了下。

在准备数据集的时候遇到了很多坑，具体的解决放在上一篇[post](https://www.wuxiaochun.cn/2018/09/17/MS-COCO-dataset/)里面了.

后来训练了模型，训练的时候正确率突然就增加到mIOU=1.0，pixel accuracy=1.0，evaluate的时候发现所有的分类都是underground，
采用自己的图片发现生成图片为全黑，于是对比MS-COCO和VOC的label，发现是准备数据集的时候出了差错，提取成了stuff segmentation的label，遂纠正。
先是训练了coco数据集中label为PASCAL VOC那20类中的图片，大概有4w张，训练出来的mIOU为69.11，
然后又训练了所有coco数据集中的图片，然后单单计算pascal那20类的得分，得到的mIOU为56.34
具体得分和效果图如下：

assessment criteria | Deeplabv3_pascal_trainaug | Deeplabv3_coco20_train | Deeplabv3_coco90_train
:--:|:--:|:--:|:--:
mIOU | 76.42 | 69.11 | 56.34
Pixel accuracy | 94.45 | 95.31 | 86.59
test_mIOU | 75.72 | - | -

![从左到右分别是原图、groundtruth、deeplabv3_pascal、deeplabv3_coco20](/img/示意图.png)

# Deeplab-v3+

## Abstract

![deeplabv3+](/img/deeplabv3+-1.jpg)
Deeplab-v3+主要使用了一种新的编码-解码架构，然后检验了Xception作为backbone使用的效果
- (a). 即Deeplab-v3的结构，使用ASPP模块获取多尺度上下文信息，然后直接上采样得到预测结果
- (b). encoder-decoder结构，高层特征提供语义，decoder逐步恢复边界信息
- (c). Deeplab-v3+结构，以Deeplab-v3为encoder，decoder结构简单
然后该论文采用了更深的Xception结构，所有的最大池化操作替换成带下采样的深度分离卷积，改进后的Xception为encode网络主体，
替换原来的ResNet101，进一步提高模型的速度和性能。

## Model Architecture

![implementation](/img/deeplabv3+-2.jpg)
可以从上图看出低级特征提供细节信息，高级特征提供语义信息。具体为什么要在图中位置融合，个人觉得是因为图片中的物体大小一般都占整个图片的1/4大小。

## realization

这里我采用了官方放出的代码，官方代码比较晦涩，github地址为：[deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab)

在PASCAL VOC trainaug和trainval上分别进行训练，得分如下：

assessment criteria | Deeplabv3_pascal_trainaug | Deeplabv3_coco20_train | Deeplabv3_coco90_train | Deeplabv3+_pascal_trainaug | Deeplabv3+_pascal_trainval
:--:|:--:|:--:|:--:|:--:|:--:
mIOU | 76.42 | 69.11 | 56.34 | 83.68 | 93.58
Pixel accuracy | 94.45 | 95.31 | 86.59 | 96.65 | 98.74
test_mIOU | 75.72 | - | - | - | -

注：因为用trainval来进行训练的deeplabv3+存在数据泄露，所以得分不具有参考性
# Stacking

在模型优化上面，我采用了stacking的方法。目前来说，有三种常见的集成学习框架：bagging，boosting和stacking。
![bagging](/img/bagging.jpg)
bagging：从训练集进行抽样组成每个基模型所需要的子训练集，对所有基模型预测的结果进行综合产生最终的预测结果

![boosting](/img/boosting.jpg)
boosting：训练过程为阶梯状，基模型按次序一一进行训练（实现上可以做到并行），基模型的训练集按照某种策略每次都进行一定的转化。对所有基模型预测的结果进行线性综合产生最终的预测结果

![stacking](/img/stacking.jpg)
stacking：用所有数据训练基模型，将训练好的所有基模型对训练集进行预测，然后将训练集的预测值作为新的训练特征，label不变，再去训练一个比较强的模型，生成预测结果

已经完成的是用三种效果较好的模型生成的图片针对每一个pixel做一个投票，来生成最好的结果
更好的思路是将三个模型输出的logits作为输入来用一个简单的三层神经网络进行训练来生成最后的logits。

assessment criteria | Deeplabv3_pascal_trainaug | Deeplabv3_coco20_train | Deeplabv3_coco90_train | Deeplabv3+_pascal_trainaug | Deeplabv3+_pascal_trainval | Stacking
:--:|:--:|:--:|:--:|:--:|:--:|:--:
mIOU | 76.42 | 69.11 | 56.34 | 83.68 | 93.58 | 87.77
Pixel accuracy | 94.45 | 95.31 | 86.59 | 96.65 | 98.74 | 97.55
test_mIOU | 75.72 | - | - | - | - | -

## DenseCRF
DenseCRF(全连接条件随机场)，是在给定一组输入随机变量条件下另外一组输出随机变量的条件概率分布模型。全连接条件随机场模型能够将邻近结点耦合，有利于将相同标记分配给空间上接近的像素。定性的说，
这些短程条件随机场主函数会清楚构建在局部手动特征上弱分类器的错误预测。具体的原理介绍见这篇[post](https://www.wuxiaochun.cn/2018/11/30/CRF/)
由于时间原因作者只做了图片中单个物体的CRF处理，如图片中有多个物体则会分类为同个物体。效果图如下：
![图1](/img/crf-1.png)
![图2](/img/crf-2.png)
![图3](/img/crf-3.png)
实现代码放github上：[semantic-segmentation](https://github.com/xiaochunWu/Semantic-Segmentation-CRF)

# To Be Continued

## Model Level
- 采用更多层次的融合，如果低级特征包含很少的语义信息，高级特征包含不够多的空间信息，那么只融合最低和最高的话效果不会达到最好。
- 我是在生成图片的基础上进行融合的，我觉得在logits的基础上融合的效果可能更好，而且可以详细设计下第二层的神经网络结构。
- 我觉得可以用两个模型来分别保留原图像的空间信息和语义信息，放一幅图来作为指引
![头脑风暴](/img/continue.jpg)
## Data Level
- 可以针对bad image和bad class增加一些图片和标注，主要是bicycle、chair、pottedplant、sofa类
- 可以对原训练集做更多augmentation的工作，比如锐化，模糊

以上。