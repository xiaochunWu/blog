---
title: TensorFlow笔记
date: 2018-06-13 10:58:53
tags: [深度学习,TensorFlow]
categories: 深度学习
---

# 神经网络
## 神经网络的基本概念
1. 张量是多维数组（列表），用‘阶’表示张量的维度
2. TensorFlow的数据类型有tf.float32、tf.int32等
3. 计算图：是承载一个或多个计算节点的一张图，只搭建网络，不运算


```python
import tensorflow as tf
x = tf.constant([[1.0, 2.0]])
w = tf.constant([[3.0],[4.0]])
y = tf.matmul(x,w)
print(y)
```

    Tensor("MatMul_3:0", shape=(1, 1), dtype=float32)
    

可以看到，print的结构显示y是一个张量，只搭建承载计算过程的计算图，并没有运算。
<!-- more -->
4. 会话：执行计算图中的节点运算
  用with结构实现，语法如下：


```python
with tf.Session() as sess:
    print(sess.run(y))
```

    [[11.]]
    

## 神经网络的搭建
1. 准备数据集，提取特征，作为输入喂给NN
2. 搭建NN结构，从输入到输出（先搭建计算图，再用会话执行）
    （NN前向传播算法-->计算输出）
3. 大量特征数据喂给NN，迭代优化NN参数
    （NN反向传播算法-->优化参数训练模型）
4. 使用训练好的模型预测和分类

### 前向传播的tensorflow描述

变量初始化、计算图节点运算都要用会话（with结构）实现

    with tf.Session() as sess:

        sess.run()
    
变量初始化：在sess.run函数中用tf.global_variables_initializer()汇总所有待优化变量

    init_op = tf.global_variables_initializer()

    sess.run(init_op)

计算图节点运算：在sess.run函数中写入待运算的节点

    sess.run(y)

用tf.placeholder占位，在sess.run函数中用feed_dict喂数据
喂一组数据：

    x = tf.placeholder(tf.float32, shape=(1,2))

    sess.run(y,feed_dict={x: [[0.5,0.6]]})

喂多组数据：

    x = tf.placeholder(tf.float32, shape=(None,2))

    sess.run(y,feed_dict={x: [[0.5,0.6]],[[0.1,0.5]],[[0.4,0.2]]，[[0.8,0.7]]})


### 反向传播的tensorflow描述

- 反向传播：训练模型参数，在所有参数上用梯度下降，使NN模型在训练数据上的损失函数最小

- 损失函数（loss）：计算得到的预测值y与已知y_的差距

- 均方误差MSE：常用的损失函数计算方法，是求前向传播计算结果与已知答案之差的平方再求平均。

    loss_mse = tf.reduce_mean(tf.square(y_ - y)) 

- 反向传播训练方法：以减小 loss 值为优化目标，有梯度下降、momentum 优化器、adam 优化器等优化方法。

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    train_step=tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)

    train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss)

- 学习率：决定每次参数更新的幅度。

## 神经网络优化

* 交叉熵(Cross Entropy)：表示两个概率分布之间的距离。交叉熵越大，两个概率分布距离越远，两个概率分布越相异；交叉熵越小，两个概率分布距离越近，两个概率分布越相似。交叉熵计算公式：𝐇(𝐲_ , 𝐲) = −∑𝐲_ ∗ 𝒍𝒐𝒈 𝒚

用 Tensorflow 函数表示为
    
    ce= -tf.reduce_mean(y_* tf.log(tf.clip_by_value(y, 1e-12, 1.0))) 
    
* softmax函数：将 n 分类的 n 个输出（y1,y2…yn）变为满足以下概率分布要求的函数。

在 Tensorflow 中，一般让模型的输出经过 sofemax 函数，以获得输出分类的概率分布，再与标准
答案对比，求出交叉熵，得到损失函数，用如下函数实现：

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    
    cem = tf.reduce_mean(ce)

* 学习率 learning_rate：表示了每次参数更新的幅度大小。学习率过大，会导致待优化的参数在最小值附近波动，不收敛；学习率过小，会导致待优化的参数收敛缓慢。在训练过程中，参数的更新向着损失函数梯度下降的方向。

参数的更新公式为：

    𝒘𝒏+𝟏 = 𝒘𝒏 − 𝒍𝒆𝒂𝒓𝒏𝒊𝒏𝒈_𝒓𝒂𝒕𝒆delta
    
* 指数衰减学习率：学习率随着训练轮数变化而动态更新

用 Tensorflow 的函数表示为：

    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(
    LEARNING_RATE_BASE,
    global_step,
    LEARNING_RATE_STEP, LEARNING_RATE_DECAY,
    staircase=True/False)
    
   注：staircase为True时，表示global_step/learning rate step取整数，学习率阶梯型衰减
   
      staircase为False时，学习率是一条平滑下降的曲线
      
* 滑动平均：记录了一段时间内模型中所有参数 w 和 b 各自的平均值。利用滑动平均值可以增强模型的泛化能力。

滑动平均值（影子）计算公式：

影子 = 衰减率 * 影子 +（1 - 衰减率）* 参数

其中，衰减率 = 𝐦𝐢𝐧 {𝑴𝑶𝑽𝑰𝑵𝑮𝑨𝑽𝑬𝑹𝑨𝑮𝑬𝑫𝑬𝑪𝑨𝒀,(𝟏+轮数)/(𝟏𝟎+轮数)}，影子初值=参数初值   

用 Tensorflow 函数表示为：

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY，global_step)
其中，MOVING_AVERAGE_DECAY 表示滑动平均衰减率，一般会赋接近 1 的值，global_step 表示当前训练了多少轮。

    ema_op = ema.apply(tf.trainable_variables())
其中，ema.apply()函数实现对括号内参数求滑动平均，tf.trainable_variables()函数实现把所有待训练参数汇总为列表。

    with tf.control_dependencies([train_step, ema_op]):
         train_op = tf.no_op(name='train') 
         
* 过拟合：神经网络模型在训练数据集上的准确率较高，在新的数据进行预测或分类时准确率较低，说明模型的泛化能力差。 

* 正则化：在损失函数中给每个参数 w 加上权重，引入模型复杂度指标，从而抑制模型噪声，减小过拟合。 

正则化计算方法：
① L1 正则化： 𝒍𝒐𝒔𝒔𝑳𝟏 = ∑𝒊|𝒘𝒊|

用 Tensorflow 函数表示:

    loss(w) = tf.contrib.layers.l1_regularizer(REGULARIZER)(w)

② L2 正则化： 𝒍𝒐𝒔𝒔𝑳𝟐 = ∑𝒊|𝒘𝒊|^𝟐

用 Tensorflow 函数表示:

    loss(w) = tf.contrib.layers.l2_regularizer(REGULARIZER)(w)

用 Tensorflow 函数实现正则化：

    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w)
    loss = cem + tf.add_n(tf.get_collection('losses')) 
    
利用L1经过训练后，会让权重得到稀疏解，即权重中的一部分项为0，这种作用相当于对原始数据进行了特征选择；利用L2进行训练后，会让权重更趋于0，但不会得到稀疏结，这样做可以避免某些权重过大；两种正则做法都可以减轻过拟合，使训练结果更加具有鲁棒性。
