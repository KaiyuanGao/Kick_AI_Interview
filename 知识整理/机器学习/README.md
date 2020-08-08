# 机器学习相关

本目录主要按照不同算法整理机器学习相关面试知识点。

- 常见的loss函数：
    - 平方损失函数（最小二乘法）
        - $L(Y, f(X))=(Y-f(X))^{2}$
    - 对数损失函数（方便极大似然估计）
        - $J(\theta)=-\frac{1}{m}\left[\sum_{i=1}^{m} y^{(i)} \log h_{\theta}\left(x^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]$`
    - 指数损失函数（Adaboost）
        -  $L(Y, f(X))=\frac{1}{n} \sum_{i=1}^{n} e^{-Y_{i} f\left(X_{i}\right)}$
    -  Hinge损失函数（SVM）
        -   $L(Y)=\max (0,1-t Y)$
    - 0-1损失
        -   $L(Y, f(X))=\left\{\begin{array}{l}{1, Y \neq f(X)} \\ {0, Y=f(X)}\end{array}\right.$
- 常见的优化算法：
    - https://blog.csdn.net/Kaiyuan_sjtu/article/details/85126721 
- 常见的激活函数：
    - sigmoid/tanh/relu/leaky relu/maxout/softmax
- 常用的评价指标：
    - precision（精确性、查准率）： 预测出的正确正样本数比上所有预测为正样本数
    - recall（召回率、查全率）:预测出的正确正样本数比上所有实际正样本数
    - F1: precision和recall的调和均值
    - ROC（Receiver Operating Characteristic）：
        - 横坐标伪正类率（False Positive Rate），预测为正但实际为负的样本占所有负样本的比例；
        - 纵坐标真正类率（True Positive Rate），预测为正且实际为正的样本占所有正样本的比例；
    - AUC：ROC曲线下的面积
- 常用的距离度量方式
    - 欧式距离：$d_{a b}=\sqrt{\left(x_{1}-y_{1}\right)^{2}+\left(x_{2}-y_{2}\right)^{2}}$
    - 曼哈顿距离：$d_{a b}=\left|x_{1}-y_{1}\right|+\left|x_{2}-y_{2}\right|$
    - 余弦相似度：$\cos \theta=\frac{a^{*} b}{|a|^{*}|b|}=\frac{x_{1} * y_{1}+x_{2} * y_{2}}{\sqrt{x_{1}^{2}+x_{2}^{2}} * \sqrt{y_{1}^{2}+y_{2}^{2}}}$
    - 切比雪夫距离：各对应坐标数值差的最大值，如平面两个向量$a=(x_{1}, y_{1}), b=(x_{2}, y_{2})$，其 $d_{a b}=\max \left\{\left|x_{1}-x_{2}\right|,\left|y_{1}-y_{2}\right|\right\}$
    - 汉明距离：字符串对应位置的不同字符个数
    - 编辑距离：两个字符串之间由一个转成另一个所需的最少编辑操作次数
    - Person相关系数：反应两个变量的线性相关性。
    - K-L散度：相对熵，$D(P \| Q)=\sum_{i=1}^{n} P(i) \log \frac{P(i)}{Q(i)}$

### 线性回归和LR

- 对于LR模型，特征有<x1,x2,...,xn> 如果手误把第一个特征又加了一次变成了n+1个特征<x1,x2,...,xn,x1>请问会有什么影响？如果是加了一个噪声特征呢？
    -  冗余特征过多，训练过程越慢
    -  会改变模型参数值，使得模型不稳定。极端假设所有特征均重复，则LR模型参数w会变为原来的1/2
- 逻辑回归损失函数为什么用极大似然？
    - 用最小二乘法目标函数就是平方损失的话是非凸函数，会有很多局部最优点 
    - 用最大似然估计目标函数就是对数似然函数
- 回归问题的损失函数都有哪些？从哪些角度设计一个合理的损失函数？
    - 绝对值损失，平方误差损失，huber损失等
    - https://www.zhihu.com/question/68390722/answer/266034620
- 逻辑回归VS线性回归？
    - 都属于广义线性模型
    - 一个回归问题一个分类问题

### gbdt、xgboost相关

- xgboost和GBDT区别：
    - 基学习模型不同：GBDT使用的是CART，而xgb还支持线性分类器
    - 优化方案不同：gbdt使用一阶导，xgb使用二阶泰勒展开
    - xgb支持自定义代价函数，但必须二阶可导
    - xgb加入了正则化项
    - 列采样
    - xgb支持缺失值的处理，会自动学习出分裂方向
    - xgb支持并行，不是tree粒度的并行而是特征粒度的并行。提前将数据进行排序存在block结构中，后面迭代直接使用减少计算量。
    - 近似直方图算法，用于高效的生成候选的分割点
    - xgb支持交叉验证，方便选取合最优参数，early stop
- xgb如何缓解过拟合？
    - 正则化：对树中叶子节点的数目L1正则，对叶子节点分数L2正则
    - 列抽样：即特征抽样
    - shrinkage：类似学习率，在学习出每一颗树之后对这个树增加一个权重参数，减小其影响力，为后续的树提供空间去优化模型
- xgb如何调参？
- xgb的特征重要性是怎么计算的？
    - 某个特征的重要性（feature score），等于它被选中为树节点分裂特征的次数的和
- adaboost流程？权重更新公式？
    - 加法模型+指数损失+前向分布算法的二类分类学习算法 
    - 流程：
        - 初始化权重分布
        - 计算基分类器在训练集上的分类误差率：$e_{m}=P\left(G_{m}\left(x_{i}\right) \neq y_{i}\right)=\sum_{i=1}^{N} w_{m i} I\left(G_{m}\left(x_{i}\right) \neq y_{i}\right)$
        - 计算该分类器的对应系数：$\alpha_{m}=\frac{1}{2} \log \frac{1-e_{m}}{e_{m}}$`
        - 更新训练集的权重分布：$w_{m+1, i}=\frac{w_{m i}}{Z_{m}} \exp \left(-\alpha_{m} y_{i} G_{m}\left(x_{i}\right)\right)$
        - 重复2-4过程直至效果比较好
        - 将所有基分类器加权求和
- gbdt流程？
    - 加法模型+负梯度函数+前向分布算法 
    - 流程：
        - 初始化基学习器：$f_{0}(x)=\arg \min _{c} \sum_{i=1}^{N} L\left(y_{i}, c\right)$
        - 对迭代轮数1,2,...,M:
            - 对所有样本，计算负梯度：$r_{i m}=-\left[\frac{\partial L\left(y_{i}, f\left(x_{i}\right)\right)}{\partial f\left(x_{i}\right)}\right]_{f=f_{m-1}}$`
            - 对$r_{i m}$拟合一个回归树，得到第m棵树的叶节点区域$R_{m j}$,  j=1,2,...,J
            - 对j=1,2,...,J：计算$c_{m j}=\arg \min _{c} \sum_{x_{i} \in R_{m j}} L\left(y_{i}, f_{m-1}\left(x_{i}\right)+c\right)$
            - 更新回归树：$f_{m}(x)=f_{m-1}(x)+\sum_{j=1}^{J} c_{m j} I\left(x \in R_{m j}\right)$
        - 得到最终回归树：$\hat{f}(x)=f_{M}(x)=\sum_{m=1}^{M} \sum_{j=1}^{J} c_{m j} I\left(x \in R_{m j}\right)$
- xgb损失函数？推导？
    -  $\mathcal{L}(\phi)=\sum_{i} l\left(\hat{y}_{i}, y_{i}\right)+\sum_{k} \Omega\left(f_{k}\right)$，其中等式右侧第二项为正则化项：$\Omega(f)=\gamma T+\frac{1}{2} \lambda\|w\|^{2}$
    -  损失函数的优化：
        -  $\mathcal{L}^{(t)}=\sum_{i=1}^{n} l\left(y_{i}, \hat{y}_{i}^{(t-1)}+f_{t}\left(\mathbf{x}_{i}\right)\right)+\Omega\left(f_{t}\right)$
        -  $\mathcal{L}^{(t)} \simeq \sum_{i=1}^{n}\left[l\left(y_{i}, \hat{y}^{(t-1)}\right)+g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\Omega\left(f_{t}\right)$
        -  $\tilde{\mathcal{L}}^{(t)}=\sum_{i=1}^{n}\left[g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\gamma T+\frac{1}{2} \lambda \sum_{j=1}^{T} w_{j}^{2}$
        -  $w_{j}^{*}=-\frac{\sum_{i \in I_{j}} g_{i}}{\sum_{i \in I_{j}} h_{i}+\lambda}$
- xgb节点分裂规则？
    - $G=\frac{1}{2}\left[\frac{G_{L}^{2}}{H_{L}+\lambda}+\frac{G_{R}^{2}}{H_{R}+\lambda}-\frac{\left(G_{L}+G_{R}\right)^{2}}{\left(H_{L}+H_{R}\right)+\lambda}\right]-\gamma$
- xgb如何处理稀疏值？
    -  会分别计算将该样本分到左右子树两种情况的增益，选择增益大的进行分裂。
    -  在测试时，如果出现缺失值，默认右子树
- 为什么gbdt不适合处理sparse特征？
    - 对于高维稀疏数据，每棵树在进行分裂时，金辉选取一个特征，大部分特征是不会被选到的
- gbdt和lr的区别？
    -  都是监督学习，判别式模型
    -  一个线性，一个非线性
    -  lr可以选择一阶导数或者二阶优化，gbdt只有一阶
    -  lr适合处理稀疏数据，gbdt适合稠密特征
    -  
- xgb的VC维？
- 为什么xgb加了二阶梯度会更好？
    - [xgboost是用二阶泰勒展开的优势在哪？](https://www.zhihu.com/question/61374305)
    - 为什么引入二阶泰勒展开？统一损失函数求导的形式以支持自定义损失函数
    - 为什么更好？类比牛顿法和SGD，二阶求导信息量更大，训练过程收敛更快更准确。
- lgb对比xgb和原始gbdt的优缺点是什么
- xgb和LR关于特征的处理有什么区别？
    - LR一般是离散值，而xgb可以连续
    - LR各个特征独立，xgb可以特征组合

### RF相关

- RF变量重要性排序原理？
    - 平均不纯度的减少：对于每棵树，按照不纯度（gini/information entropy等)给特征排序，然后整个森林取平均
    - 平均准确率的减少：测量每种特征对模型预测准确率的影响

### kmeans相关

- KNN和kmeans的区别？
    - KNN是有监督学习（分类/回归），kmeans是无监督学习（聚类）
    - k的含义：KNN中k表示找到距离样本X最近的k个点；kmeans表示人为定义将数据分为k个类别
- kmeans的基本流程：
    - 选取k个类的初始中心点
    - 对所有点进行计算距离划分到最近的类
    - 重新计算每一类的中心点
    - 重新回到上述2，如果中心点没变则输出类别
- k值的选取？
    - 先验知识
    - Elbow method
- 如何选取初始中心点？
    - https://www.cnblogs.com/pinard/p/6164214.html   
- kmeans时间复杂度和空间复杂度？
    -  时间：O(NlogN)
    -  空间： O(K*(M+N))

### EM HMM CRF相关

- EM算法推导？jensen不等式确定的下界？
    - EM算法就是含有隐变量的概率模型参数的极大似然估计
    - ![](http://ww1.sinaimg.cn/large/afd47e42ly1g26n7hu2r0j20f60pegoo.jpg)
    - Jensen不等式：
        - 对于凸函数：$E[f(X)] \geq f(E[X])$
        - 对于凹函数：$E[f(X)] \leq f(E[X])$
    - EM算法是收敛的，但是不能保证收敛到全局最优 
    - 对初始值的选取敏感
- HMM和CRF的区别？https://www.zhihu.com/question/53458773
    - 两者都可以用于序列模型
    - CRF是无向图，HMM是有向图
    - CRF是判别式模型，HMM是生成式模型
    - CRF没有假设，HMM有马尔科夫假设
    - CRF可以全局最优，HMM可能局部最优
- CRF模型优化目标，怎么训练的？
    - CRF的三个问题以及解决思路
- HMM做了哪些独立性假设？
    -  有限历史假设：即当前状态仅仅与前一个状态有关
    -  输出独立假设：即输出状态仅仅与当前的隐状态有关
    -  齐次性假设：状态与时间无关
- viterbi算法原理
    - 动态规划求最大路径

### 决策树相关

- 信息增益、信息增益比、基尼系数的公式和原理
    - 信息增益：g=H(D)-H(D|A), 在特征选择时偏向于取值较多的特征,对应ID3算法，该算法只有树的生成，容易过拟合；
    - 信息增益率：$g_{\mathrm{g}}(D, A)=\frac{g(D, A)}{H_{A}(D)} \quad, \quad \mathrm{H}_{A}(D)=-\sum_{i=1}^{n} \frac{\left|D_{\mathrm{i}}\right|}{|D|} \log _{2} \frac{\left|D_{\mathrm{i}}\right|}{|D|}$
    - gini指数：$\operatorname{Gini}(p)=\sum_{k=1}^{K} p_{k}\left(1-p_{k}\right)=1-\sum_{k=1}^{K} p_{k}^{2}$
- 决策树的VC维
    - VC维是描述模型复杂度的，模型假设空间越大，vc维越高
    - http://www.flickering.cn/machine_learning/2015/04/vc%E7%BB%B4%E7%9A%84%E6%9D%A5%E9%BE%99%E5%8E%BB%E8%84%89/
    - VC = 节点数+1
- 决策树怎么做特征离散化？
    - 可以采用二分法对连续属性离散化：$T_{a}=\left\{\frac{a^{i}+a^{i+1}}{2} | 1 \leq i \leq n-1\right\}$
- 决策树的缺失值怎么处理？
    - 对于建树节点分裂过程缺失：对特征计算非缺失样本的熵然后乘上权重（非缺失样本占比）就是该特征最终的熵 
    - 对于建树完成训练时缺失某个特征：将样本分配到每颗分裂出的子树中，然后乘上落入该子树的概率（即该子树中样本比上总样本）
    - 对于预测过程：确定额的划分
- CART决策树的剪枝？
    - https://www.zhihu.com/question/22697086
- CART回归树是怎么做节点划分的？
    - 采用启发式的方法 
- CART为什么采用gini指数作为特征划分标准？
    - 信息增益(比)是基于信息论为熵模型的，会涉及大量的对数运算。而基尼系数和熵之半的曲线非常接近，都可以近似代表分类误差率

### SVM相关

- SVM推导
- SVM损失函数？
    - hinge：$L(y)=\max (0,1-t \cdot y)$
    - 表示当样本点被分类正确且函数间隔大于1时，损失为0；否则损失为$1-t \cdot y$
- 为什么要使用hinge loss？
    - 只考虑支持向量的影响
- SVR原理
- 核函数原理、哪些地方引入、如何选择？
    - $\mathrm{K}(\mathrm{x}, \mathrm{z})=<\Phi(\mathrm{x}), \Phi(\mathrm{Z})>$
    - https://www.zhihu.com/question/24627666
    - 核函数的作用就是一个从低维到高维的映射
    - 线性：$K\left(v_{1}, v_{2}\right)=<v_{1}, v_{2}>$
    - 多项式：$K\left(v_{1}, v_{2}\right)=\left(\gamma<v_{1}, v_{2}>+c\right)^{n}$
    - RBF：$K\left(v_{1}, v_{2}\right)=\exp \left(-\gamma\left\|v_{1}-v_{2}\right\|^{2}\right)$
    - sigmoid：$K\left(v_{1}, v_{2}\right)=\tanh \left(\gamma<v_{1}, v_{2}>+c\right)$
    - 如果特征维数很大（跟样本数量差不多），优先选用线性；如果特征数量小，样本数量一般，选用高斯核；如果特征数量小，样本数量很大，手工添加一些feature变成第一种情况。
- 为什么需要转成对偶形式？
    - 原问题是一个凸二次规划问题   
    - 优化了复杂度。由求特征向量w转化为求比例系数
    - 可以方便引出核函数
- 线性回归的梯度下降和牛顿法求解公式的推导
- 最速下降法和共轭梯度法 wolfe条件 最速下降法和共轭梯度法的收敛速度如何判断
    - 最速下降法即梯度下降：一阶信
    - 共轭梯度法：介于最速下降和牛顿法之间的一种优化方法，仅需要一阶导数信息（方向限制在初始点的共轭区间内），但收敛速度较快，同时避免了牛顿法求Hessian矩阵的缺点
    - wolfe条件：跟line search有关
- LDA中有哪些分布？定义？什么是共轭分布？
    - 共轭分布：在贝叶斯统计中，如果先验分布和后验分布属于同一类分布，则成这俩为共轭分布
    - LDA过程：
        - 从狄利克雷分布D(a)中采样生成文档的主题分布$\theta_{i}$
        - 从主题$\theta_{i}$的多项式分布中采样生成文档第j个词的主题$z_{i, j}$
        - 从狄利克雷分布D(b)中采样生成主题$z_{i, j}$对应的词语分布$\phi_{\tilde{z}_{i, j}}$
        - 从词语的多项式分布中采样生成最终词语$w_{i, j}$
- k折交叉验证中k取值？
    - 从偏差和方差角度回答： 
        - 当k取值很小时，比如k=2，此时模型训练数据较少，不容易拟合正确，所以偏差较高，方差较低
        - 当k取值较大时，比如k=n，此时相当于所有数据都用于训练，容易过拟合，所以偏差低，方差高
    - 论文给出k值参考公式：k=log(N)，N为样本总数
- KKT条件？（L为拉格朗日函数，g(x), h(x)为约束函数）
```math
$$\left\{\begin{array}{l}{\nabla_{x} L=0} \\ {\mu g(x)=0} \\ {h(x)=0} \\ {g(x) \leq 0} \\ {\lambda \neq 0} \\ {\mu \geq 0}\end{array}\right.$$
```

### 降维算法相关

- 常见降维方法：L1，PCA, LDA t-SNE
- 什么是主成分？
- PCA是一种无监督的降维方法，为了让映射后的样本具有最大的发散性（即尽可能少的重叠，保留原有信息）；
- LDA是一种有监督的降维方法，为了让映射后的样本具有最好的分类性能（即类内方差最小，类间方差最大）
- 局部线性嵌入(LLE)是一种非线性降维算法，能够使降维后的数据较好地保持原有流形结构。

##### 各种ML模型比较
- 逻辑回归VS线性回归？
    - 都属于广义线性模型
    - 一个回归问题一个分类问题
- SVM和LR逻辑回归？
    - 都属于线性分类算法，判别模型，监督学习
    - 损失函数不同
    - 确定决策边界时考虑的训练点不同
    - SVM有核函数，LR虽然也可以用但是一般不适用
    - SVM自带正则项

### 其他

- 谈谈牛顿法？牛顿法如何优化的？
    - 牛顿法迭代公式：$x_{k+1}=x_{k}-\frac{f^{\prime}\left(x_{k}\right)}{f^{\prime \prime}\left(x_{k}\right)}$,优点是二阶导收敛速度快，但是需要就算hessian矩阵的逆，计算复杂
    - 优化：拟牛顿法，使用正定矩阵来近似Hessian矩阵的逆
- 交叉熵损失函数公式？怎么推导得到的？
    - 公式： $L=\sum_{i=1}^{N} y^{(i)} \log \hat{y}^{(i)}+\left(1-y^{(i)}\right) \log \left(1-\hat{y}^{(i)}\right)$
    - 推导：
        - $P(y=1 | x)= \hat{y}$
        - $P(y=0 | x)=1- \hat{y}$
        - $P(y|x)=\hat{y}^{y} \cdot(1-\hat{y})^{1-y}$
        - log化：$\log P(y | x)=\log \left(\hat{y}^{y} \cdot(1-\hat{y})^{1-y}\right)=y \log \hat{y}+(1-y) \log (1-\hat{y})$
        - 求log化的最大值，前面加个负号就是求最小值，就是交叉熵的公式了
- mapreduce原理
     - map和reduce两个过程：分而治之
- 共线性的特征会对模型产生怎样的影响？
    -  LR中特征强相关，不会影响最优性，但是会造成权重的数值解不稳定性（即模型系数每次都不一样）
- 朴素贝叶斯公式，先验概率，后验概率，条件概率
    - 贝叶斯公式：$P\left(Y | X\right)=\frac{P\left(X | Y\right) P\left(Y\right)}{ P(Y)}$
- 各种机器学习的应用场景分别是什么？例如，k近邻,贝叶斯，决策树，svm，逻辑斯蒂回归和最大熵模型。
    - https://www.zhihu.com/question/26726794 
- 如何解决L1不可导问题？
    - L1能产生稀疏解，而且稀疏解的泛化能力比较好
    - subgradient: 绝对值函数只有在零点是不可导的，可以把abs的导数定义成符号函数sgn
    - proximal gradient：
- L0，L1，L2正则化
    - L0正则化的值是模型参数中非零参数的个数，L0很难优化求解是NP难问题
    - L1对应拉普拉斯分布
    - L2对应高斯分布
- 哪些常用的分类器是有VC维的，怎么计算？
    - 线性分类器： d+1
    - 高斯核分类器： 无穷
    - 神经网络：参数数量
    - 决策树：节点数+1
- 特征选择方法？
    - 皮尔森相关系数：
        - 协方差和标准差比值：$\rho_{X, Y}=\frac{\operatorname{cov}(X, Y)}{\sigma_{X} \sigma_{Y}}$
        - 衡量两个变量之间的线性关系，取值[-1,1]，对非线性有明显缺陷
    - 卡方检验
        - 表示自变量对应变量的相关性：$\chi^{2}=\sum \frac{(A-E)^{2}}{E}$
    - 互信息
        - $I(X ; Y)=\sum_{x \in X} \sum_{y \in Y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)}$
    - 基于惩罚项的特征选择
        -  L1正则
    - 基于学习模型的特征排序
        - RF，GBDT，xgboost
- 最大似然估计(MLE)VS最大后验估计(MAP)
    - MLE： $\hat{\theta}_{\mathrm{MLE}}=\arg \max P(X |  \theta)$，但由于连乘会造成浮点下溢，通常使用最大化对数形式
    - MAP: $\hat{\theta}_{\mathrm{MAP}}=\underset{\theta}{\operatorname{argmax}} P(\theta | X)=\underset{\theta}{\operatorname{argmax}} \frac{P(X | \theta) P(\theta)}{P(X)} \propto \operatorname{argmax}_{\theta} P(X | \theta) P(\theta)$
    - MLE是频率派的思想，认为参数$\theta$是固定的；而MAP是贝叶斯派的思想，认为参数符合某种概率分布（先验概率）
    - MLE可以认为是经验风险最小化，MAP可以认为是结构风险最小化
- 判别式模型 VS 生成式模型？
    - 判别式，无向图，求解的是条件概率，如LR,SVM,NN,CRF等
    - 生成式，有向图，求解的是联合概率，如HMM,NB,LDA等
    - 由生成式模型可以得到判别式模型，但反之不行
- 模型融合？原理？怎么选融合的模型？
    -  模型融合的方法有：voting/averaging/bagging/boosting/stacking等
    -  stacking融合：交叉验证+拼接；https://blog.csdn.net/u011630575/article/details/81302994
    -  融合模型要求： 好而不同。要求模型效果优秀且各模型个体之间尽量不同（如模型类型，模型超参数等）

