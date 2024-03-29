---
title: Batch Normalization
date: 2018-08-20 21:54:04
tags: [深度学习,BN]
categories: 深度学习
---
Batch Normalization作为最近一年来DL的重要成果，已经广泛被证明其有效性和重要性。

机器学习领域有个很重要的假设：IID独立同分布假设，就是假设训练数据和测试数据是满足相同分布的，这是通过训练数据获得的模型能够在测试集
获得好的效果的一个基本保障。那BatchNorm的作用是什么呢？BatchNorm就是在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布的。
<!-- more -->
## 一、"Internal Covariate Shift"问题

首先说明Mini-batch SGD相对于One Example SGD的两个优势：
- 梯度更新方向更准确
- 并行计算速度快

那么所谓的covariate shift就是 如果ML系统实例集合<X,Y>中的输入值X的分布老是变，就不符合IID假设，神经网络模型很难稳定的学到规律。

对于深度学习这种包含很多隐层的网络结构，在训练过程中，因为各层参数不停在变化，所以每个隐层都会面临covariate shift的问题，也就
是在训练过程中，隐层的输入分布老是变来变去，这就是所谓的“Internal Covariate Shift”，Internal指的是深层网络的隐层，是发生在网络内部
的事情，而不是covariate shift问题只发生在输入层。

之前的研究表明如果在图像处理中对输入图像进行白化（Whiten）操作的话——所谓白化，就是对输入数据分布变换到0均值，单位方差的正态分布——那么神经网络会较快收敛。
那么图像作为深度神经网络的输入层，通过白化能够加快收敛，对于深度网络来说，其中某个隐层的神经元是下一层的输入，可以把BN理解成为对深层神经网络每个隐层神经元的
激活值做简化版本的白化操作。

## 二、 BatchNorm的本质思想

因为深层神经网络在做非线性变换前的激活输入值（就是那个x=WU+B，U是输入）随着网络深度加深或者在训练过程中，其分布逐渐发生偏移或者变动，之所以训练收敛慢，一般是整体
分布逐渐往非线性函数的取值区间的上下限两端靠近（对于Sigmoid函数来说，意味着激活输入值WU+B是大的负值或正值），所以这导致反向传播时低层神经网络的梯度消失，
这是训练深层神经网络收敛越来越慢的本质原因，而BN就是通过一定的规范化手段，把每层神经网络任意神经元这个输入值的分布强行拉回到均值为0方差为1的标准正态分布，
其实就是把越来越偏的分布强制拉回比较标准的分布，这样使得激活输入值落在非线性函数对输入比较敏感的区域，这样输入的小变化就会导致损失函数较大的变化，意思是这样
让梯度变大，避免梯度消失问题产生，而且梯度变大意味着学习收敛速度快，能大大加快训练速度。

总而言之，言而总之：对于每个隐层神经元，把逐渐向非线性函数映射后向取值区间极限饱和区靠拢的输入分布强制拉回到均值为0方差为1的比较标准的正态分布，
使得非线性变换函数的输入值落入对输入比较敏感的区域，以此避免梯度消失问题。

## 三、 训练阶段如何做BatchNorm

假设对于一个深层神经网络来说，其中两层结构如下：

![png](/img/bn-1.png)

要对每个隐层神经元的激活值做BN，可以想象成每个隐层又加上了一层BN操作层，它位于X=WU+B激活值获得之后，非线性函数变换之前，其图示如下：

![png](/img/bn-2.png)

对于Mini-Batch SGD来说，一次训练过程里面包含m个训练实例，其具体BN操作就是对于隐层内每个神经元的激活值来说，进行如下变换：

![png](/img/bn-3.png)

要注意，这里t层某个神经元的x(k)不是指原始输入，就是说不是t-1层每个神经元的输出，而是t层这个神经元的线性激活x=WU+B，这里的U才是t-1层神经元的输出。
变换的意思是：某个神经元对应的原始的激活x通过减去mini-Batch内m个实例获得的m个激活x求得的均值E(x)并除以求得的方差Var(x)来进行转换。

经过这个变换后某个神经元的激活x形成了均值为0，方差为1的正态分布，目的是把值往后续要进行的非线性变换的线性区拉动，增大导数值，增强反向传播信息流动性，
加快训练收敛速度。但是这样会导致网络表达能力下降，为了防止这一点，每个神经元增加两个调节参数（scale和shift），这两个参数是通过训练来学习到的，用来对变换后的激活反变换，
使得网络表达能力增强，即对变换后的激活进行如下的scale和shift操作，这其实是变换的反操作：

![png](/img/bn-4.png)

BN其具体操作流程，如论文中描述的一样：

![png](/img/bn-5.png)


## 参考文献：
- 《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》
- [深入理解Batch Normalization批标准化](https://www.cnblogs.com/guoyaohua/p/8724433.html)
