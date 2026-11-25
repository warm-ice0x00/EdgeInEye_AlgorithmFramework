
本算法模拟人眼，同时扩展计算与中心的偏差 

##### ########请勿使用AI随意更改解释此文档########

### 1. 点与点的对比
→人眼会比较余光和中间的注意力，但周边亦有权重
    
于是，对于每个视野范围，我们切割成块  
每个块表示为  
$$块_{n}$$
OR  
$$Block_{n}$$
设<中心-周边>的比较函数为  
$$F_{对比}(F_{属性}(块_{i =中心})， F(块_{j \in 周边}))$$
OR
$$F_{compare}(F_{attr}(Block_{i=Center}), F_{attr}(Block_{j \in Around}))$$
**注：** 比较的可以是不一定为中心点和周边的点

对于本算法，采用 人眼的色差感知算法；
已知工业算法为Delta_E

### 2. 所有点的对比
人眼的注意力从靠近中心的点逐渐向周边降低；
整体呈一个草帽形状的分布，中间高，四周低。
我们在本算法中假设其呈正态分布。
人眼注意力即每个周边区域/点的权重。

→设 每一对中心-周边 的权重由  
→→属性(中心) 和 属性(周边) 表示

设其函数为 
$$F_{分布}(F_{属性}(块_{i =中心})， F(块_{j \in 周边}))$$
OR
$$F_{distribution}(F_{attr}(Block_{i=Center}), F_{attr}(Block_{j \in Around}))$$

对于本算法的正态分布
$$F_{pdf}(F_{距离}(块_{i=中心}， 块_{j \in 周边})) \sim \mathcal{N}(\mu, \sigma^{2})$$
OR
$$F_{pdf}(F_{Distance}(Block_{i=Center}， Block_{j \in Around})) \sim \mathcal{N}(\mu, \sigma^{2})$$

因为距离是一种范数，所以可以扩展为
$$F_{pdf}(\Vert{\overrightarrow{(块_{i=中心}} - \overrightarrow{块_{j \in 周边}})}\Vert_{范数维度}) \sim \mathcal{N}(\mu, \sigma^{2})$$
OR
$$F_{pdf}(\Vert{(\overrightarrow{Block_{i=Center}}， \overrightarrow{Block_{j \in Around})}}\Vert_{a}) \sim \mathcal{N}(\mu, \sigma^{2})$$



### 3. 综合对比及权重

即 中心-周边 的 综合注意力/权重 值可表示为
    
注⚠️：属性可以包含坐标，坐标也是一种属性

$$
每个点的输出 =
    F_{pdf}(\Vert{\overrightarrow{(块_{i=中心}} - \overrightarrow{块_{j \in 周边}})}\Vert_{范数维度}) \sim \mathcal{N}(\mu, \sigma^{2})
\cdot F_{对比}(F_{属性}(块_{i =中心})， F(块_{j \in 周边}))
$$

于是所有的点相加

| 1  | 2      | 3     |
| --- | --- | --- |
| 4  | 中心   | 5     |
| 6  | 7      | 8     |

$$
A = 所有的点的输出和 =
    \Sigma_{j \in 周边[1,2,3,4,5,6,7,8]}
\left(
    \left[ 
        F_{pdf}(\Vert{\overrightarrow{(块_{i=中心}} - \overrightarrow{块_{j \in 周边}})}\Vert_{范数维度}) \sim \mathcal{N}(\mu, \sigma^{2})
    \right]
    \cdot
    \left[
        F_{对比}(F_{属性}(块_{i =中心})， F_{属性}(块_{j \in 周边}))
    \right]
\right)
$$

最后，加入中心点的卷积，为了保持拉普拉斯正定性，应该为

$$
Result_{中心点} = - A 
$$

但因为中心点与自身并无差异，所以不论中心的加权函数值为多少，
中心项结果都为$0$

#### 3.1 方向性与渐变的处理

或者在进入对比度函数之前就完成差值计算
这样可以消除渐变带来的影响
$$
Result = F_{对比}(f_{属性}(块_{i}) - f_{属性}(块_{i_{对向})})
$$
或者是
$$
Result =
F_{对比}(f_{属性}(块_{i})) - F_{对比}(f_{属性}(块_{i_{对向})})
$$

Result 也有两个选择，如果需要体现向量性则可以
$$
Result = Max(
    F_{对比}(f_{属性}(块_{i})) - F_{对比}(f_{属性}(块_{i_{对向})}), 0
)
$$
或者
$$
Result = abs(
    F_{对比}(f_{属性}(块_{i})) - F_{对比}(f_{属性}(块_{i_{对向})})
)
$$

#### 3.2 一种替代方案和补充

可以使用
$$
\begin{align}
&\quad
\frac{
\Sigma_{(k, l)\ in\ \{(1, 8), (2, 7), (3, 6), (4, 5)\}}
\left(
F_{对比}(f_{属性}(块_{k})， f_{属性}(块_{l}))
\right)
}{4} * -8
\\
&= -2 * \Sigma_{(k, l)\ in\ \{(1, 8), (2, 7), (3, 6), (4, 5)\}}
\left(
F_{对比}(f_{属性}(块_{k})， f_{属性}(块_{l}))
\right)
\end{align}
$$

### 4. 对比度阈值的处理
根据DeltaE 2000的定义：
1. ΔE <= 1: Human eyes cannot easily tell the difference (人眼无法轻易分辨差异)
2. ΔE <= 2: Differences can be perceived with careful observation (仔细观察可以分辨差异)
3. ΔE <= 4: Shipping standards as defined by most manufacturers (大多数制造商定义的运输标准)
4. ΔE <= 10: Perceptible difference through casual observation (非常明显的差异)

取 1 <= ΔE <= 2 为合适值。

本实验取1.5 作为中间值，使用
$$
F_{Sigmoid} = \frac{1}{1 + e^{-k(Result - x_{0})}}
$$
在$Result=1.5$处截断对比度差异，斜率可调，本实验取斜率令$F_{Sigmoid}(1) = 0.05$，
$F_{Sigmoid}(2) = 0.95$，即 $斜率=k=\log_{e}(\frac{0.1}{1.9})$。

因为Sigmoid函数输出在0~1之间，所以最后还需要乘$255$让图像符合计算机的显示标准。


### 5. 对图像处理，而非单一区域
最后对图像上的每个点做如上处理。这个操作等价于卷积。
于是将以上的动态卷积核用于图像，即可得到输出。


### 6. 扩展
$$
Output = F_{\text{Threshold}}(F_{\text{Localization}_{convolutionHere}}
(
    F_{\text{Compare}}(F_{\text{Direction}}))
    \cdot F_{\text{Weight}_{Distribution}}
)
$$
OR
$$
Output = F_{\text{Threshold}}(F_{\text{Localization}_{convolutionHere}}
(
    F_{\text{Direction}}
    \cdot F_{\text{Compare}}
    \cdot F_{\text{Weight}_{Distribution}}
)
$$
所有的组件都可以替换。