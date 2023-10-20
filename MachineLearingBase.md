#名词解释
##参数和超参数
超参数决定参数
* parameter : w, b
* hyperparameter: learning rate $a$, hidden layer L, interations(梯度下降法循环的数量)， hidden units n^[1]^, activation function

###要不停地尝试来寻找“最优值”
由于GPU、CPU或Data等可能会发生的变化，“最优值”会发生改变，需要通过各种检验方法寻找当下的“最优值”。

##训练集Train、验证集Dev、测试集Test sets
* 要确保训练集和测试集相匹配，即让它们==出自于同一分布==（网页或个人上传）。
* 当数据非常庞大时，测试集和验证集可以设置得非常小。
* 当不需要做**无偏评估**时，可以不设置测试集，此时验证集也就是“测试集”。

##方差variance和偏差bias
* variance ： Train set error 和 Dev set error 相差很大
* bias ：（整体）误差很大

选择规模更大的网络提升效果
通过正则化减小方差或使用更多数据
***保证方差或偏差一方减小时另一方不会受到太大的不良影响***

---
**2023/10/19**
