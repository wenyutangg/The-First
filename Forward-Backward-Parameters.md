#深度神经网络
输入特征： x = a^[0]^  
预测输出： a^[L]^ = $\widehat{y}$

##矩阵的维数
>向量化->列向量组合  

“权重矩阵的行数对应于当前层的神经元数量，列数对应于上一层的神经元数量。”  
z^[L]^ = w^[L]^x + b^[L]^  
*  z : (n^[L]^,1) 
*  w : (n^[L]^,n^[L-1]^)
*  b : (n^[L]^,1) 
*  x : (n^[0]^,1)

>经过Python Broadcast之后

*  Z , A, dZ, dA : (n^[L]^,m) 
*  W : (n^[L]^,n^[L-1]^)
*  b : (n^[L]^,m) 
*  X : (n^[0]^,m)   ==(A^[0]^ = X )==

##本质（？）
layer1:==特征==探测 （边缘） "small"
layer2:各部分 "larger"
layer3:组合

*简单*——>*复杂*

##Forward and backward functions
|Functions|Input|Output|cache|
|:----:|:----:|:----:|:----:|
Forward|a^[L-1]^|a^[L]^|z^[L]^
Backward|da^[L]^|da^[L-1]^,dw^[L]^,db^[L]^|z^[L]^

>*“复杂的是数据，而不是代码。”*
---
**2023/10/19**