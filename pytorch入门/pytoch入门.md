## 线性模型

**机器学习的工作流程**

首先收集数据集，然后训练模型，然后预测和推理。

![image-20240814150120927](pytoch入门.assets/image-20240814150120927.png)

**监督学习**

监督学习是一种机器学习方法，它通过分析带有已知输出的示例数据集来训练模型，以便能够预测或决定未见过数据的输出。

![image-20240814150519899](pytoch入门.assets/image-20240814150519899.png)

**过拟合**：模型在训练集上表现很好，但在测试集中表现较差，即泛化能力差。

![image-20240814151318328](pytoch入门.assets/image-20240814151318328.png)

通常会把训练集分成，训练集和开发集，开发集通常用于模型性能的评估。



**模型设计**：

要解决的问题：**对数据而言，什么样的模型是最合适的？？？**即f ( x )的形式是什么？？？
最基本的模型是线性模型。

线性模型中，训练时关键的就是确定w和b的值。w被称为权重，b称为偏置。

**对模型做一个简化，去掉截距b。*y*^表示预测结果。**

![image-20240814154641899](pytoch入门.assets/image-20240814154641899.png)

权重可能大可能小，不一定正好落在真实值，因此需要进行评估。

**评估模型（loss）**

![image-20240814154940145](pytoch入门.assets/image-20240814154940145.png)

**如何找到合适的权重值，使得损失最小 ？？？**

损失函数（Loss function）是针对一个样本的，对于整个训练集需要将每个样本的预测值和真实值求差然后计算均方根误差。

![image-20240814155254387](pytoch入门.assets/image-20240814155254387.png)

![image-20240814155433918](pytoch入门.assets/image-20240814155433918.png)

**然后将均方差以图表的形式呈现**

![image-20240814155603141](pytoch入门.assets/image-20240814155603141.png)

可以得出w=2的均方差是最小的。

**实践代码**

```python
import numpy as np
import matplotlib.pyplot as plt

# 训练集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def virtual(w_r, x):
    return w_r * x


def loss(y_r, y):
    return (y_r - y) * (y_r - y)


# w_list存储每个权重值，mse_list存储每个权重值对应的损失值
w_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):

    print(f"w = {w}")
    loss_sum = 0

    for x_real, y_real in zip(x_data, y_data):
        # zip(numbers, letters)创建一个生成 (x, y) 形式的元组的迭代器，[(numbers[0], letters[0]),…,(numbers[n], letters[n])]
        y_virtual = virtual(w, x_real)
        loss_val = loss(y_real, y_virtual)

        loss_sum += loss_val  # 叠加每个样本的损失值
        print(x_real, y_real, y_virtual, loss_val)

    print(loss_sum / 3)
    w_list.append(w)
    mse_list.append(loss_sum / 3)
    print()

# 画图 x轴是权重w，y轴是loss值，即表示每个权重值w对应的loss值
import matplotlib.pyplot as plt
plt.plot(w_list, mse_list)
plt.ylabel("loss")
plt.xlabel("w")
plt.show()
```



**运行结果**

![image-20240815174041300](pytoch入门.assets/image-20240815174041300.png)

补充：在深度学习中做训练的时候，loss曲线中一般不是用权重来做横坐标，而是训练轮数（epoch）



