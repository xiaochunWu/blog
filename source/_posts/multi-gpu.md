---
title: multi-gpu
date: 2018-11-01 16:38:13
tags: [深度学习, 模型构建, tricks]
categories: 深度学习
---
# multi-gpu 的原理
数据并行的原理很简单，如下图，其中CPU主要负责梯度平均和参数更新，而GPU1和GPU2主要负责训练model replica。
<!-- more -->
![训练步骤如图](/img/multi-gpu.png)
1. 在GPU1、GPU2上分别定义模型参数变量，网络结构
2. 对于单独的GPU，分别从数据管道读取不同的数据块，然后做forward propagation来计算出loss，再计算在当前variables下的gradients
3. 把所有GPU输出的梯度数据转移到CPU上，先进行梯度取平均操作，然后进行模型参数的更新
4. 重复1-3，直到模型收敛

在1中定义模型参数时，要考虑到不同model replica之间要能够share variables，因此要采用tf.get_variable()函数而不是直接tf.Variables()。另外，因为tensorflow和theano类似，都是先定义好tensor Graph，再基于已经定义好的Graph进行模型迭代式训练的。因此在每次迭代过程中，只会对当前的模型参数进行更新，而不会调用tf.get_variable()函数重新定义模型变量，因此变量共享只是存在于模型定义阶段的一个概念。

在实际使用过程中我发现其实不用在模型构建的时候考虑并行架构也可以实现multi-gpu，因为google有放出multi-gpu的api，详情见[multi-gpu](https://github.com/tensorflow/models/blob/master/research/slim/deployment/model_deploy.py)

# Tensorflow切换使用CPU/GPU

```python
	config = tf.ConfigProto(device_count={'GPU': 0}) # 0表示使用CPU, 1则是GPU
```
或者修改model_deploy.DeploymentConfig(clone_on_cpu=True)

# Tensorflow多GPU训练

在/models/research/slim/depolyment/model_deploy.py中，有说明：
> DeploymentConfig parameters: 
> * num_clones: Number of model clones to deploy in each replica. 
> * clone_on_cpu: True if clones should be placed on CPU. 
> * replica_id: Integer. Index of the replica for which the model is deployed. Usually 0 for the chief replica. 
> * num_replicas: Number of replicas to use. 
> * num_ps_tasks: Number of tasks for the ps job. 0 to not use replicas. 
> * worker_job_name: A name for the worker job. 
> * ps_job_name: A name for the parameter server job.

Example:

```python
  config = model_deploy.DeploymentConfig(
      num_clones=FLAGS.num_clones,
      clone_on_cpu=FLAGS.clone_on_cpu,
      replica_id=FLAGS.task,
      num_replicas=FLAGS.num_replicas,
      num_ps_tasks=FLAGS.num_ps_tasks)	
```

**在terminal还是要设置CUDA_VISIBLE_DEVICES=(),要不会占用剩下GPU的显存。
