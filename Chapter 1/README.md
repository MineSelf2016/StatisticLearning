# 第 1 章 统计学习方法概论

## 1.1 统计学习的定义
<p>
统计学习也称为统计机器学习。
</p>

统计学习是关于计算机基于数据构建概率统计模型并运用模型对数据进行预测与分析的一门学科。

### 1. 统计学习的特点

<ul>
    <li>统计学习以计算机及网络为平台，是建立在计算机及网络之上的；
    <li>统计学习以数据为研究对象，是数据驱动的学科；
    <li>统计学习的目的是对数据进行预测与分析；
    <li>统计学习以方法为中心，统计学习方法构建模型并应用模型进行预测与分析；
    <li>统计学习是概率论、统计学、信息论、计算理论、最优化理论及计算机科学等多个领域的交叉学科，并且在发展中逐步形成独自的理论体系与方法论。
</ul>

### 2. 统计学习的对象
<p>
<b>数据</b>
</p>

<p>
统计学习关于数据的基本假设是同类数据具有一定的<font color="red">统计规律性</font>，这是统计学习的前提。例如，可以用随机变量描述数据中的特征，用概率分布描述数据的统计规律。
</p>

<p>
在统计学习过程中，以变量或变量组表示数据。数据分为由连续变量和离散变量表示的类型。
</p>

### 3. 统计学习的目的
<p>统计学习用于对数据进行预测与分析，特别是对未知新数据进行预测与分析。对数据的预测可以使计算机更加智能化，或者说使计算机的某些性能得到提高；对数据的分析可以让人们获取新的知识，给人们带来新的发现。</p>

### 4. 统计学习的方法
<p>统计学习的方法是基于数据构建统计模型从而对数据进行预测与分析。统计学习由监督学习（supervised learning）、非监督学习（unsupervised learning）、半监督学习（semi-supervised learning）和强化学习（reinforcement learning）等组成。</p>

<p>统计学习方法的定义：从给定的、有限的、用于学习的<b>训练数据（training data）</b>集合出发，假设数据是独立同分布产生的；并且假设要学习的模型属于某个函数的集合，称为<b>假设空间（hypothesis space）</b>；应用某个<b>评价准则（evaluation criterion）</b>，从假设空间中选取一个最优的模型，使他对已知训练数据及未知测试数据（test data）在给定的评价准则下有最优的预测；最优模型的选取由算法实现。</p>

<p>统计学习方法包括模型的假设空间、模型选择的准则以及模型学习的算法，称其为统计学习方法的三要素，简称为模型（model）、策略（strategy）和算法（algorithm）</p>

实现统计学习方法的步骤如下：
<ul>
    <li>得到一个有限的训练数据集合；
    <li>确定包含所有可能的模型的假设空间，即学习模型的集合；
    <li>确定模型选择的准则，即学习的策略；
    <li>实现求解最优模型的算法，即学习的算法；
    <li>通过学习方法选择最优模型；
    <li>利用学习的最优模型对新数据进行预测或分析。
</ul>

<p>监督学习方法，主要包括用于<b>分类</b>、<b>标注</b>与<b>回归</b>问题的方法。这些方法在自然语言处理、信息检索、文本数据挖掘等领域中有着极其广泛的应用。</p>

### 5. 统计学习的研究
<p>统计学习研究一般包括统计学习方法、统计学习理论及统计学习应用三个方面。</p>

### 6. 统计学习的重要性
<p>近20年来，统计学习无论是在理论还是在应用方面都得到了巨大的发展，有许多重大突破，统计学习已被成功地应用到人工智能、模型识别、数据挖掘、自然语言处理、语音识别、图像识别、信息检索和生物信息等许多计算机应用领域中，并且成为这些领域的核心技术。</p>

<p>统计学习是计算机科学发展的一个重要组成部分。可以认为计算机科学由三维组成：<b>系统、计算、信息</b>。统计学习主要属于信息这一维，并在其中起着核心作用。</p>

