---
title: Semantic Segmentation
date: 2018-09-03 21:06:12
tags: [语义分割,深度学习]
categories: 深度学习
---
最近在做语义分割的研究，就上周汇报做一个简单记录。

# What exactly is Semantic Segmentation?

所谓语义分割，就是在像素水平上理解一幅图，例如我们想要把图片中的每一个像素都给分类。

通过语义分割，我们可以把一幅图片中的不同类物体识别出来，并可以区分出边界，从而可以实现像素级别的预测
<!-- more -->
# What are the different approacher?

* CNN：CNN来实现语义分割主要是通过对一个像素在图像中的patch来进行分类。对于CNN来说，直到第一个全连接层
之前，输入图片的大小可以是不固定的，但是有了全连接层之后，就要求输入大小保持一致。所以会有以下缺点：
1. 存储开销大 
2. 计算效率低
3. patch的大小限制了感受野的大小

* FCN：将CNN的全连接层替换成卷积层就变成了全卷积神经网络，这也是现在语义分割领域大部分架构的基础结构。

* encoder-decoder architecture：因为池化层的使用丢失了位置信息，使得分类结果不能够实现绝对匹配。

![png](/img/u-net.png)

* dilated/atrous convolutions：膨胀卷积可以不通过池化获得较大的感受野，减小信息损失。

* CRF：条件随机场预处理常用于改善分割效果，它是一种基于底层图像像素强度进行平滑分割的图模型，工作原理是
灰度相近的像素易被标注为同一类别，通常可令分值提高1-2%。

# Summaries of previous works

## FCN

- End to end convolutional networks
- Deconvolutional layers
- Skip connections

![png](/img/FCN.png)

FCN是将全连接层改为卷积层之后实现端到端的卷积网络，它应用了反卷积层来进行上采样。由于池化过程造成信息丢失，
上采样生成的分割图较为粗糙，所以从高分辨率的特征图引入跳跃连接来改善上采样的粗糙程度。

## Dilated Convolutions

- Dilated convolutions
- Context module

除了全连接层，使用卷积神经网络来实现分割的另一个主要问题是池化层，池化层扩大了感受野，有效继承了上层信息，
但是同时丢弃了位置信息，然而，语义分割的要求是对于分类结果的绝对匹配，所以提出了膨胀卷积。
膨胀卷积能够极大的扩大感受野同时不减小空间维度。
Context module级联了不同膨胀系数的膨胀卷积层，输入输出同尺寸，能够提取不同规模特征中的信息，得到更精确的分割结果。

## DeepLab(v1&v2)

- ASPP
- Atrous convolutions
- CRF

![png](/img/deeplabv1.png)

这是google推出的一系列架构，这两个版本较相似，就放在一块说了。
该架构是将原始图像的多个重新缩放版本传递到CNN的并行分支(图像金字塔)中，或者使用采样率不同的多个并行空洞卷积层(ASPP)，
实现多尺度处理，膨胀卷积不再另行介绍。
全连接CRF可以实现结构化预测，该部分的训练/微调需作为后处理的步骤单独进行。

## RefineNet

- Encoder-Decoder architecture
- residual connection design

![png](/img/refinenet.png)

上文提到的为了解决池化层而提出的膨胀卷积也有弊端，其需要大量高分辨率特征图，计算成本高，占用内存大。
所以提出了RefineNet，它主要由三个部分构成：
1. 不同尺度（也可能只有一个输入尺度）的特征输入首先经过两个Residual模块的处理；
2. 之后是不同尺寸的特征进行融合。当然如果只有一个输入尺度，该模块则可以省去。所有特征上采样至最大的输入尺寸，然后进行加和。
上采样之前的卷积模块是为了调整不同特征的数值尺度；
3. 最后是一个链式的pooling模块。其设计本意是使用侧支上一系列的pooling来获取背景信息（通常尺寸较大）。直连通路上的ReLU可以在
不显著影响梯度流通的情况下提高后续pooling的性能，同时不让网络的训练对学习率很敏感。
最后再经过一个Residual模块即得RefineNet的输出。

## PSPNet

- Pyramid pooling module
- Use auxiliary loss

![png](/img/PSPNet.png)

该架构采用4层金字塔模块，该模块将ResNet的特征图与并行池化层的上采样输出结果连接起来，从而保留了位置信息。
另外，在主分支损失之外增加了附加损失，能够更快更有效的调校模型。

## DeepLab v3

- Improved ASPP
- Atrous convolutions in cascade

![png](/img/deeplabv3.png)

在该架构中，具有不同atrous rates的ASPP能够有效的捕获多尺度信息。不过，随着采样率的增加，有效特征区域会逐渐变小。
当采用具有不同atrous rates的3x3 filter应用到65x65 featuremap上时，在rate值接近于feature map大小的极端情况，该3x3
 filter不能捕获整个图像内容信息，而退化成了一个简单的1x1 filter，因为只有中心filter权重才是有效的。所以最后要对特征
进行双线性上采样到特定的空间维度。
该架构中的级联模块的逐步翻倍的atrous rates和ASPP模块增强图像级的特征，探讨了多采样率和有效感受野下的滤波器特性。

## UperNet

- Unified Perceptual Parsing
- Broden+

![png](/img/UperNet.png)

该架构要实现的是一个全新概念，统一感知解析(Unified Perceptual Parsing),即让机器系统尽可能多的识别出一副图像中的视觉概念，
所以重新构建了数据集Broden+。
为了使深度卷积网络的感受野足够大，本架构将PSPNet中的PPM用于骨干网络的最后一层。因为图像级信息更适合场景分类，PPM 模块之后的特征图被用来对scene分类
。来自 FPN 的所有层相融合的特征图被用来对Object和Part进行分类。FPN 中带有最高分辨率的特征图用来对Material进行分类。纹理特征是最简单的特征，最容易发现和辨别
，因此Texture 被附加到 ResNet 中的 Res-2 模块，并在整个网络完成其他任务的训练之后进行优化。

## DeepLab v3+

- New encoder-decoder architecture
- Xception

![png](/img/deeplabv3+_0.png)

该模型主要使用了一种全新的编码-解码架构，然后探索了Xception和深度分离卷积在模型上的使用
(a): 即DeepLabv3的结构，使用ASPP模块获取多尺度上下文信息，直接上采样得到预测结果
(b): encoder-decoder结构，高层特征提供语义，decoder逐步恢复边界信息
(c): DeepLabv3+结构，以DeepLabv3为encoder，decoder结构简单

![png](/img/deeplabv3+_1.png)

![png](/img/deeplabv3+_2.png)

模型改进了MSRA的Xception工作，采用了更深的Xception结构，不同的地方在于不修改entry flow network的结构，这样可以快速计算和有效的使用内存。
所有的最大池化操作替换成带下采样的深度分离卷积，这能够应用扩张分离卷积扩展feature的分辨率。
在每个3×33×3的深度卷积后增加BN层和ReLU。
改进后的Xception为encode网络主体，替换原本DeepLabv3的ResNet101，进一步提高模型的速度和性能。

# Datasets comparison

![png](/img/datasets.png)

> 最近在mentor的指导下尝试复现deeplab v3，之后如果复现成功的话会把github放上来~

以上。


