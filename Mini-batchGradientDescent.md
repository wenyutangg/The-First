#Mini-batch gradient descent
当数据集非常巨大时，把训练集分成一些小的子集baby training set——mini-batches。
例如：
原数据集|mini-batch
:---:|:---:|
X = (n~x~,m)|X^{t}^ = (n~x~,512)
Y = (1,m)|Y^{t}^=(1,512)
>(i) : 第i个训练样本
[l] : 神经网络层数
{t} : mini-batch

mini-batch中，处理过程最大的变化就是X^{t}^，Y^{t}^。
![Alt text](<mini-batch gradient descent.png>)
*前向传播和反向传播过程*

cost图像走势向下，但噪声会增多（例如X^{2}^，Y^{2}^比较难算，必须处理一些misalable样本,成本更高）。
##mini-batch size
size|name|feature
:---:|:---:|:---:|
m|batch|耗时长
1|stochastic|效率低

->随机梯度下降法：有很多噪声，永远不会收敛，只会在最小值附近波动（可以通过降低学习率减小噪声）。

![Alt text](<the differences.png>)
###mini-batch的优点
1. 通过大量向量化，处理样本速度加快。
2. 不需要等它训练完就可以开始进行后续操作。

*如果出问题，可以缓慢减小学习率。*

###其他
* 当训练集较小(m<=2000)时，一般使用batch gradient decent批量梯度下降算法。
* 一般把mini-batch size 取为2的幂，typical size = (64,128,256,512),它也是一个超参数。通过多次尝试找到最合适的size。
* 要确保所有的X^{t}^，Y^{t}^可以放进CPU/GPU内存里。

---
**2023/10/26**