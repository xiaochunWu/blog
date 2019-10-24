---
title: Introduction to ML with Python
date: 2018-06-05 16:37:39
tags: [机器学习,特征处理]
categories: 机器学习
---

# sklearn中的交叉验证
## 标准交叉验证


```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
```


```python
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logerg = LogisticRegression()

scores = cross_val_score(logerg, iris.data, iris.target)
print "Cross-validation scores: {}".format(scores)
```

    Cross-validation scores: [ 0.96078431  0.92156863  0.95833333]
    


```python
scores = cross_val_score(logerg, iris.data, iris.target, cv=5)
print "Cross-validation scores: {}".format(scores)
```

    Cross-validation scores: [ 1.          0.96666667  0.93333333  0.9         1.        ]
    


```python
print "Average cross-validation score: {:.2f}".format(scores.mean())
```

    Average cross-validation score: 0.96
<!-- more -->    

## 对交叉验证的更多控制


```python
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)
```


```python
print "Cross-validation scores:\n{}".format(
       cross_val_score(logerg, iris.data, iris.target, cv=kfold))
```

    Cross-validation scores:
    [ 1.          0.93333333  0.43333333  0.96666667  0.43333333]
    


```python
kfold = KFold(n_splits=3)
print "Cross-validation scores:\n{}".format(
       cross_val_score(logerg, iris.data, iris.target, cv=kfold))
```

    Cross-validation scores:
    [ 0.  0.  0.]
    


```python
kfold = KFold(n_splits=3, shuffle=True, random_state=0)
print "Cross-validation scores:\n{}".format(
       cross_val_score(logerg, iris.data, iris.target, cv=kfold))
```

    Cross-validation scores:
    [ 0.9   0.96  0.96]
    

## 留一法交叉验证


```python
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
scores = cross_val_score(logerg, iris.data, iris.target, cv=loo)
print "Number of cv iterations: ", len(scores)
print "Mean accuracy: {:.2f}".format(scores.mean())
```

    Number of cv iterations:  150
    Mean accuracy: 0.95
    

## 打乱划分交叉验证


```python
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logerg, iris.data, iris.target, cv=shuffle_split)
print "Cross-validation scores:\n{}".format(scores)
```

    Cross-validation scores:
    [ 0.97333333  0.97333333  0.92        0.88        0.97333333  0.96
      0.94666667  0.96        0.77333333  0.89333333]
    

## 分组交叉验证


```python
from sklearn.model_selection import GroupKFold
from sklearn.datasets import make_blobs
# 创建模拟数据集
X, y = make_blobs(n_samples=12, random_state=0)
# 假设前3个样本属于同一组，接下来的4个样本属于同一组，以此类推
groups = [0,0,0,1,1,1,1,2,2,3,3,3]
scores = cross_val_score(logerg, X, y, groups, cv=GroupKFold(n_splits=3))
print "Cross-validation scores:\n{}".format(scores)
```

    Cross-validation scores:
    [ 0.75        0.8         0.66666667]
    

# 网格搜索
## 简单网格搜索


```python
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=0)
print "Size of training set: {}  size of the test set: {}".format(
    X_train.shape[0], X_test.shape[0])

best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma,C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}
            
print "Best score: {:.2f}".format(best_score)
print "Best parameters: {}".format(best_parameters)
```

    Size of training set: 112  size of the test set: 38
    Best score: 0.97
    Best parameters: {'C': 100, 'gamma': 0.001}
    

- training set: Model fitting
- validation set: Patameter selection
- test set: Evaluation


```python
import mglearn
mglearn.plots.plot_grid_search_overview()
```


```python
from sklearn.metrics import precision_recall_curve
X, y = make_blobs(n_samples=450, n_features=2, centers=2, cluster_std=[7.0, 2],
                random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=.05).fit(X_train, y_train)
precision, recall, thresholds = precision_recall_curve(
    y_test, svc.decision_function(X_test))
```


```python
mglearn.plots.plot_decision_threshold()
```


```python
from sklearn.metrics import classification_report
print classification_report(y_test, svc.predict(X_test))
```

                 precision    recall  f1-score   support
    
              0       0.96      0.80      0.87        60
              1       0.81      0.96      0.88        53
    
    avg / total       0.89      0.88      0.88       113
    
    

# 预处理与缩放
## 应用数据变换


```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                   random_state=1)
print(X_train.shape)
print(X_test.shape)    
```

    (426L, 30L)
    (143L, 30L)
    


```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
MinMaxScaler(copy=True, feature_range=(0,1))
#变换数据
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

# 降维、特征提取和流形学习

## PCA


```python
from  sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
print "Original shape: {}".format(X_scaled.shape)
print "Reduced shape: {}".format(X_pca.shape)
```

    Original shape: (569L, 30L)
    Reduced shape: (569L, 2L)
    

## 非负矩阵分解 NMF


```python
import mglearn
mglearn.plots.plot_nmf_illustration()
```


```python
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
X_people = people.data[mask]
y_people = people.target[mask]
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
nmf = NMF(n_cpmponents=15, random_state=0)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

fix, axes = plt.subplots(3, 5, figsize=(15,12),
                        subplot_kw={'xticks':(), 'yticks':()})
for i, (component,ax) in enumerate(zip(nmf.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape))
    ax.set_title("{}.component".format(i))
```

## 用t-SNE进行流形学习


```python
from sklearn.datasets import load_digits
digits = load_digits()

fig, axes = plt.subplots(2, 5, figsize=(10,5),
                        subplot_kw={'xticks':(), 'yticks':()})
for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)
```

# 数据表示与特征工程

## 分箱、离散化、线性模型与树


```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import mglearn
import numpy as np
import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
plt.plot(line, reg.predict(line), label="decision tree")

reg = LinearRegression().fit(X, y)
plt.plot(line, reg.predict(line), label="linear regression")

plt.plot(X[:,0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
```




    <matplotlib.legend.Legend at 0x103c4da0>



## 自动化特征选择

### 单变量统计


```python
citibike = mglearn.datasets.load_citibike()
print "Citi Bike data:\n{}".format(citibike.head())
```

    Citi Bike data:
    starttime
    2015-08-01 00:00:00     3.0
    2015-08-01 03:00:00     0.0
    2015-08-01 06:00:00     9.0
    2015-08-01 09:00:00    41.0
    2015-08-01 12:00:00    39.0
    Freq: 3H, Name: one, dtype: float64
    


```python
import pandas as pd
plt.figure(figsize=(10, 3))
xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(),
                      freq='D')
plt.plot(citibike, linewidth=1)
plt.xlabel('Date')
plt.ylabel('Rentals')
```




    Text(0,0.5,u'Rentals')



# 算法链与管道

## 举例说明信息泄露

考虑一个假象的回归任务，包含从高斯分布中独立采样的100个样本和10000个特征。还从高斯分布中对响应进行采样：


```python
import numpy as np
rnd = np.random.RandomState(seed=0)
X = rnd.normal(size=(100,10000))
y = rnd.normal(size=(100,))
```

考虑到创建数据集的方式，数据X与目标y之间没有任何关系，所以应该不可能从这个数据集中学到任何内容。现在首先利用SelectPercentile特征选择从10000个特征中选择信息量最大的特征，然后使用交叉验证对Ridge回归进行评估：


```python
from sklearn.feature_selection import SelectPercentile, f_regression

select = SelectPercentile(score_func=f_regression, percentile=5).fit(X, y)
X_selected = select.transform(X)
print "X_selected.shape:{}".format(X_selected.shape)
```

    X_selected.shape:(100L, 500L)
    


```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
print "Cross-validation accuracy(cv only on ridge):{:.2f}".format(np.mean(cross_val_score(Ridge(), X_selected, y, cv=5)))
```

    Cross-validation accuracy(cv only on ridge):0.91
    

交叉验证计算得到的平均R^2为0.91，表示这是一个非常线性的模型，但是这明显不对，因为数据是完全随机的。这里的特征选择从10000个随机特征中（碰巧）选出了与目标相关性非常好的一些特征。由于我们在交叉验证之外对特征选择进行拟合，所以能够找到在训练部分和测试部分都相关的特征。从测试部分泄露出去的信息包含的信息量非常大，导致得到非常不切实际的结果。将上面的结果和正确的交叉验证（使用管道）进行对比：


```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([("select", SelectPercentile(score_func=f_regression,
                                            percentile=5)),
                ("ridge",Ridge())])
print "Cross-validation accuracy (pipeline):{:.2f}".format(
        np.mean(cross_val_score(pipe, X, y, cv=5)))
```

    Cross-validation accuracy (pipeline):-0.25
    

这一次得到了负的R^2分数，表示模型很差。利用管道，特征选择现在位于交叉验证循环内部，也就是说，仅适用数据的训练部分来选择特征，而不使用测试部分。特征选择找到的特征在训练集中与目标相关，但由于数据是完全随机的，这些特征在测试集中并不与目标相关。在这个例子中，修正特征选择中的数据泄露问题，结论也由“模型表现很好”变为“模型根本没有效果”。
