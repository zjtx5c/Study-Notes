# pytorch 框架精摘

## `torch.nn`

目前暂定学习 `Containers`，`Linear Layers`， `Loss Functions`, `Transform_layers` 等模块...

### `Buffer` 与 `Parameter`

#### [`Buffer`](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Buffer.html#torch.nn.parameter.Buffer)

* 在 PyTorch 中，`buffer` 是指**不会被优化（即不会被 `optimizer` 更新），但仍然会随模型保存和加载的Tensor**（会被 `state_dict()` 记录）。它是Tensor的子类。它适用于那些在前向传播中需要使用，但不需要梯度更新的变量，例如移动平均参数、归一化层的统计信息、一些常数等。

  * 普通 Tensor 直接赋值为 `Module` 的属性时，不会被加入 buffer 列表，亦不会进入计算图。
  * 搞懂 普通`Tensor(nn.Parameter)` 与 普通 `Tensor(非nn.Parameter)` 以及 `buffer` 的区别
    * 是否被视为参数模型
    * 是否会计算梯度
    * 在 `state_dic` 中的存储

* `register_buffer()` 的使用

  * 在 `torch.nn.Module` 中，可以使用 `register_buffer(name, tensor)` 方法添加一个 buffer。

  * 一般是在类的初始化模块中使用 `self.regiser_buffer(name, tensor)` 来创建一个 buffer tensor.其中`.register_buffer()` 是**继承**的来自 `nn.Module()` 的方法

    * 验证是否会计算梯度 `model.my_buffer.requires_grad`

    * 验证是否会被模型存储 `model.state_dict()`

    * 查看其所在的设备 `model.my_buffer.device`

    * 访问模块中所有 `buffer` 
      ```python
      for name, buf in model.named_buffers():
          print(name, buf.shape)
      ```

#### [`Parameter`](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html)

* **`Parameter`** 类是 `Tensor` 的一个子类，它有一个非常特殊的属性：当它作为 `Module` 的属性赋值时，它会自动被添加到 `Module` 的 **参数列表** 中，并且会出现在 `parameters()` 迭代器中，进而让优化器能更新这些参数（==因为它本质上是 `Tensor` 的子类，但专门用于 `torch.nn.Module` 以区分 **可训练参数** 和普通 `Tensor`。所以这个操作才感觉特殊==）。与此不同的是，**普通的 `Tensor`** 并没有这种行为，直接赋值给 `Module` 的属性时，**不会** 自动被加入到参数列表中。

* 了解其主要特点

  * 可学习参数
    * 通过 `nn.Parameter` 创建的参数默认会在模型的 `parameters()` 方法中。（前提是需要将它赋值给 `nn.Module` 的属性，否则将不会被 `parameters()` 识别，也不会被优化器更新。）
  * 自动注册
    * 当你把 `nn.Parameter` 放在 `nn.Module` 的 `__init__` 方法中时，它会自动被注册为模型的一个参数。
  * 具有梯度

* 何时使用 `nn.Parameter`

  * 定义神经网络的权重和偏置或其他可学习参数
  * 在模型外部定义可学习参数

* 其他用法

  * `model.parameters()`： 返回的是一个包含所有 `nn.Parameter` **对象**的迭代器。

  * 获取模型中所有 `nn.Parameter` 对象的参数
    ```python
    for param in model.parameters():
        print(param)
    ```

### `Containers`

#### [`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

* 所有神经网络模块的**基类**。我们模型也应该继承这个类。`Module` 还可以包含其他 `Module`，允许它们嵌套在树结构中。并在 `__init__` 方法中定义层结构，在 `forward` 方法中定义数据的前向传播过程。

  * ```python 
    import torch.nn as nn
    class A(nn.Module):
        def __init__(self):
            super(A, self).__init__()
            
    a = A()
    ```

    我们向上述一样这样定义了一个类，那么实例 `a` 就是一个 `nn.Module`可以像 PyTorch 其他模型一样使用，例如添加层、进行前向传播等。

* 核心作用

  * 管理模型参数（`nn.Parameter`）
    * 自动注册，在`.parameters()`中返回所有可训练参数
  * 管理子模块
    * `nn.Module` 允许你嵌套使用多个 `nn.Module`，例如在定义神经网络时，可能会使用多个 `nn.Linear`、`nn.Conv2d` 等层。
    * PyTorch 会**自动递归地管理子模块中的参数**，方便保存和加载模型。
  * 提供 `forward` 方法
    * 你需要在子类中实现 `forward()`，定义数据如何通过网络进行前向传播。
  * 提供模型参数的存取方法
    * `.parameters()` 获取所有可训练参数（`nn.Parameter`）
    * `.buffers()` 获取所有 `register_buffer()` 注册的 `Tensor`
    * `.state_dict()` / `.load_state_dict()` 用于存储和加载模型参数

* 如何理解 `nn.Module` 递归管理子模块？

  * 在 PyTorch 中，`nn.Module` 允许我们将一个模型拆分成多个子模块（例如线性层、卷积层等），然后再将这些子模块组合成一个更大的模型。PyTorch **会自动递归管理这些子模块**，包括：

    - **自动注册子模块**：当你在 `nn.Module` 中定义子模块（例如 `self.layer = nn.Linear(...)`），它们会被自动添加到 `model.children()` 或 `model.named_children()` 中。

      - 可以使用 `.children()` 查看其子模块 / `print(list(model.children()))` 
      - 可以看 `jupyternotebook` 笔记

    - **递归访问子模块的参数**：`model.parameters()` 会自动**递归**收集所有子模块的参数，而无需手动添加。

      - ```python
        for name, param in model.named_parameters():
            print(name, param.shape)
        ```

    - **自动保存和加载模型参数**：使用 `state_dict()` 和 `load_state_dict()` 时，所有子模块的参数都会被递归管理，保存和恢复时无需手动处理。

* 常用的 `Module` 功能

  | 方法                 | 作用                                                      |
  | -------------------- | --------------------------------------------------------- |
  | `parameters()`       | 获取所有可训练参数                                        |
  | `named_parameters()` | 获取带名称的参数（用过这个方法debug，检查过梯度断裂问题） |
  | `children()`         | 获取直接子模块                                            |
  | `named_children()`   | 获取直接子模块及名称                                      |
  | `modules()`          | 获取所有子模块（包括嵌套）                                |
  | `named_modules()`    | 获取所有子模块及名称                                      |
  | `state_dict()`       | 获取模型的状态字典                                        |
  | `load_state_dict()`  | 加载模型状态                                              |
  | `register_buffer()`  | **注册不参与优化的 `Tensor` （居然可以直接在外面实现）**  |
  | `train(mode=True)`   | 设定训练模式                                              |
  | `eval()`             | 设定评估模式，会关闭 Dropout 及 BatchNorm 统计更新。      |
  | `apply(fn)`          | 对所有子模块应用函数（常用于初始化参数）                  |
  | `zero_grad()`        | 清空所有参数的梯度                                        |

#### [Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)

* `nn.Sequential` 是 PyTorch 中用于构建**顺序模型**的容器，它按照**给定顺序**将一系列层组合成一个网络。使用它可以简化模型定义，使代码更加简洁。

  ```python
  model = nn.Sequential(
      nn.Linear(20, 10),
      nn.ReLU(),
      nn.Sequential(
          nn.Linear(10, 5),
          nn.ReLU()
      ),
      nn.Linear(5, 1)
  )
  
  input = torch.Tensor(5, 20)
  res = model(input)
  ```

  

* 可以自定义层名称与嵌套

* 适用场景：多层感知机、CNN、**快速构建简单网络**、**省略 `forward()` 方法**（就是一进一出简单又直接）

* 注意事项：

  * **无法处理多个输入**，比如 GNN 可能需要 `forward()` 处理多个输入，需要手写 `nn.Module` 继承类。

    * `nn.Sequential` 不能直接用于处理多个输入，这是显然的。
      * 我甚至立马能想象出来它的 `forward` 是怎么处理的
    
    **不适用于具有条件逻辑的模型**，如 `if` 语句或跳跃连接（Residual Networks 需要自定义 `forward()`）。



#### [ModuleList](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)

* 在列表中保存子模块
* `nn.ModuleList` **可以像 Python 普通列表那样索引和遍历，其中的 `nn.Module` 子模块会被 PyTorch 正确注册**，因此它们可以被 `model.parameters()` 访问，并被优化器更新。
* 它的一些方法
  * `.append()`——追加一个 `nn.Module`（在末尾）
    * `layers.append(nn.Linear(20, 5))`

  * `.extend()`——扩展多个 `nn.Module`
    * `layers.extend([nn.Linear(5, 3), nn.Sigmoid()])`

  * `.insert()`——在指定索引插入 `nn.Module`
    * `layers.insert(1, nn.BatchNorm1d(20))  # 在索引 1 插入 BatchNorm 层`


#### [ModuleDict](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html)

* 在字典中保存子模块
* 同上
* 它的一些方法
  * `.clear()` `.items()` `.keys()` `.pop(key)` `.values()`
  * `.update(modules)`：它允许你通过将另一个 `ModuleDict` 或者字典类型的数据合并到当前的 `ModuleDict` 中来更新现有的子模块。
* 应用场景：以后见到了再好好感受下

### Transformer（搁置）



### Linear Layers

#### [`nn.Identity`](https://pytorch.org/docs/stable/generated/torch.nn.Identity.html#torch.nn.Identity)

`nn.Identity` 是 PyTorch 中的一个**伪层（identity layer）**，它不执行任何数学变换，直接返回输入数据。这是一个非常简单但非常有用的模块，通常在模型设计、调试或动态网络结构中作为**占位符**使用。

* 理解它的作用与占位符
* 基本功能
  * **输入输出相同**：`nn.Identity()` 层对输入不做任何修改，直接返回原始数据。
  * **无参数学运算**：它没有任何可学习的参数（权重或偏置），计算开销几乎为零。
  * **类似于 `lambda x: x`**，但它是 `nn.Module` 的子类，可以像其他 PyTorch 层一样被添加到模型中。

* 用处
  * **占位符（最有用的）**： `self.act = nn.ReLU() if use_activation else nn.Identity()  # 动态选择是否使用激活`
  * 调试模型：插入 `Identity` 检查输入流

#### [`nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

* 对传入的数据进行放射变换： $y = xW^{\top} + b$

  - 输入 $x$ 的形状：`(*, in_features)`

  - 权重 $W$ 的形状：`(out_features, in_features)`

  - 偏置 $b$ 的形状：`(out_features)`（可设置 true or false）

  - 输出 $y$ 的形状：`(*, out_features)`
    （`*` 表示任意额外的维度，如 batch_size）
  - **因此，我们的这个操作 `nn.Linear(in_feats, out_feats)` 定义的其实是 $W^{\top}$ 而非 $W$**，这是 PyTorch 设计中的一个关键细节，容易让人产生困惑。

* 关键细节

  * 权重初始化：默认使用均匀分布 $\mathcal{U}(-\sqrt k, \sqrt k), k = 1 / in\_feates$，其中 
  * 若禁用偏置，则为 $y = xW^{\top}$
  * 高维处理：`nn.Linear` 会自动支持更高维度的输入（如 3D 张量），**仅对最后一维做线性变换**

* 思考与 `torch.matmul()` 的区别

  | 操作         | `nn.Linear`                        | `torch.matmul()`         |
  | :----------- | :--------------------------------- | :----------------------- |
  | **功能**     | 封装了权重和偏置的线性变换         | 纯矩阵乘法               |
  | **偏置**     | 支持（可选）                       | 需手动添加               |
  | **梯度计算** | 自动跟踪 `weight` 和 `bias` 的梯度 | 需手动定义可训练参数     |
  | **适用场景** | 模型构建时的标准层                 | 自定义低级运算或特殊结构 |

  * 思考：实现一下两者的等效实现，尤其是高维度的情况（理解广播机制）
    ```python
    import torch
    import torch.nn as nn
    linear = nn.Linear(100, 50)
    x = torch.randn(10, 20, 100)
    y1 = linear(x)
    WT = linear.weight.transpose(0, 1).unsqueeze(0)
    # b = linear.bias.unsqueeze(dim = 0).unsqueeze(dim = 0)
    b = linear.bias.view(1, 1, -1)
    y2 = torch.matmul(x, WT) + b
    torch.allclose(y1, y2, atol=1e-6)
    ```
  
    ==注意：这里需要设置浮点数误差为 `1e-6`，才能返回为 `True`，精度再高一点则是 `False`==！这个BUG修了好久，但是如果不自己广播，让系统自动广播，那么精度几乎是无限接近的就很奇怪。。

## `torch.Tensor`

直接看 [这篇](https://pytorch.zhangxiann.com/1-ji-ben-gai-nian/1.2-tensor-zhang-liang-jie-shao) 与 [这篇](https://pytorch.zhangxiann.com/1-ji-ben-gai-nian/1.3-zhang-liang-cao-zuo-yu-xian-xing-hui-gui)

### 概念

张量是包含单一数据类型元素的**多维矩阵**，它是标量、向量、矩阵的高维扩展。。是 `pytorch` 中最基本的数据结构！

* 张量的优势：

  * **支持 GPU 加速**（可以用 `.to('cuda')` 将张量转移到 GPU）。

    **支持自动微分**（在深度学习中用于计算梯度）。

    **支持广播机制**（不同形状的张量可以自动扩展计算）。

* 拥有**八个**属性

  * `data`, `grad`, `grad_fn`,  `requires_grad`,  `is_leaf`,  `dtype`, `shape`, `device`

* `Tensor` 创建的方法

  * 直接创建 `Tensor`： `torch.tensor()`

  * 根据数值创建 `torch.zeors()`,  `torch.zeros_like(input)`: 创建与 `input` **形状相同** 的全零张量
    `torch.ones()`, `torch.ones_like()`
    `torch.fill((3, 3), 2)`, `torch.fill_like(input, val)`: 创建自定义类型的张量

    `torch.arange(start = 0, end, step = 1)`

    `torch.linspace(start, end, steps = 100)`: `stpes` 表示数列长度（元素个数）:功能：创建均分的 1 维张量。数值区间为 [start, end]
    `torch.logspace(start, end, stpes = 100, base = 10)`\

    `torch.eye()`

  * 根据概率创建  `Tensor()`

    * `torch.normal()`: `torch.normal(mean, std, *, generator=None, out=None)`

      功能：生成正态分布 (高斯分布)

    * `torch.randn(size)` 与 `torch.randn_like(input)`
      功能：生成标准正态分布。

    * `torch.rand(size) 和 torch.rand_like(input)`

      功能：在区间 `[0, 1)` 上生成均匀分布。

    * `torch.randint(low = 0, high, size)` 和 `torch.randint_like(input, low = 0, high)`

    * `torch.randperm()`

      功能：生成从` 0` 到 `n-1` 的随机排列。常用于生成索引。

      

### 张量的操作

直接看[这篇](https://pytorch.zhangxiann.com/1-ji-ben-gai-nian/1.3-zhang-liang-cao-zuo-yu-xian-xing-hui-gui)，不想写了

* 彻底理解拼接操作的机制，**自己的感悟是从括号中认识维度（`dim`）**

  * 尤其是：`dim = -1`  是**最内层里面的==元素（是单个值了，没有被其他括号包围）==**
    `dim = 0` 是 最**外层里面的元素（一般情况下并非单个值，仍然被其他括号包围）**
    * 对应的元素要==一一对应（括号对括号，值对值）==！
    * 记忆法，`dim` 指定哪个维度，哪个维度就会变化
    * 不要以为 `dim` 小的元素数量就比 `dim` 大的元素数量大，它们都有各自存在的对应区域！！（好好悟一下！），其**索引**能够“穿越”其他维度。
  * `torch.stack()` 的机制？先增一维！

* 切分操作（鸽了）

* 索引操作

* 变换操作

  * `torch.reshape()`
    功能：变换张量的形状。当张量**在内存中是连续**时，返回的张量和原来的张量共享数据内存，改变一个变量时，另一个变量也会被改变。

    * 应用场景

      **数据展平**（Flatten），如 `torch.reshape(x, (-1,))`。

      **调整形状以适配模型输入**，如 `torch.reshape(x, (batch_size, channels, height, width))`。

      **与 NumPy 的 `.reshape()` 类似，适用于改变张量形状但不改变数据本身**。

    * 提醒：**新维度应该有实际的物理或语义上的==意义==**，否则可能会导致数据在计算过程中被错误使用。

      在改变形状时，确保数据排列不会影响计算逻辑。

      如果只是为了处理维度，可以用 `permute()` 或 `view()`，而不是盲目 `reshape()`。

  * `torch.transpose(input, dim0, dim1)`
    功能：交换张量的**两个维度**。常用于图像的变换，比如把`c*h*w`变换为`h*w*c`。也会配合广播机制**一起使用**

    * `input`: 要交换的变量

      `dim0`: 要交换的第一个维度

      `dim1`: 要交换的第二个维度

  * `torch.t(input)`

    * 二维矩阵的转置

  * `torch.squeeze(input, dim)`

    * 压缩长度为1的维度

    * 应用场景：

      **去掉多余的维度**，尤其是在处理神经网络的输出时。

      **匹配输入格式**，有些 API 可能要求特定维度的张量。

  * `torch.unsqueeze(input, dim)`

    * 功能：根据 dim 扩展维度，长度为 1。

    * 应用场景：

    * **扩展维度以匹配操作**，如在批量处理中手动添加 batch 维度 (`dim=0`)。

      **增加通道维度**，如将 `(H, W)` 变成 `(1, H, W)` 以适配 CNN 输入。

### 张量的数学运算

`torch.matmul()`： 会自动运用广播机制



### 对张量维度的深入理解

* `(B, num_reg, d)` 这一个问题，最初遇到是在学习 `deepwalk` 中的 `skip_gram` 中，其定义负样本嵌入的维度是 `embed_neg.shape()` 为 `(B, num_neg, d` 表示这一批次的**样本数量**（中心词）为 `B`，每个中心词采样 `num_neg` 个数量的**负样本**，每个负样本的词嵌入维度为 `d`。
  在定义时我们这样定义： `neg = np.random.randint(0, 100, size=(10, 5))`。这表明：样本数为10，每个中心词采样 5 个负样本(但是可能会重复)。如果是这样的话，我认为在处理上 第 1 个中心词 是不能对上第 2 个中心词所对应的 5 个负样本（这里的维度其实是有冗余的）。**事实上，维度确实是冗余的但这种设计是为了让每个中心词有自己独立的一组负样本。**最终经过处理得到 `neg_logits = torch.matmul(embed_src, embed_neg.transpose(1, 2))`        `(B2, B1, num_neg)`
  gpt的理解：

  > 由于 `neg_logits` 形状 `(B, B, num_neg)`，它的索引意义是：
  >
  > - `neg_logits[i, j, k]` 表示 **第 `i` 个中心词** 和 **第 `j` 个中心词的 `k` 号负样本** 之间的相似度。
  >
  > 这其实是不必要的，因为：
  >
  > - **本来每个中心词 `i` 只需要对自己的负样本 `neg[i]` 进行计算，而不是和其他中心词的负样本 `neg[j]` 进行计算**。
  > - 这样计算出来的 `neg_logits` 可能包含 **不相关的信息**。

  ==这个先搁了吧，还是没有理解很透彻==

### 其他问题

* `torch.tensor` 和 `torch.Tensor` 的区别是什么？

  * `torch.tensor` 是用于创建张量的**函数**，并且会根据输入创建一个数据副本，它需要**显示**地提供数据`data`。
  * `torch.Tensor` 是**张量的类**构造函数，**通常**会返回一个未经初始化的张量（它其实是 `torch.FloatTensor` 的别名）。

  一般来说，建议使用 `torch.tensor` 来显式地创建张量，因为它的行为更加明确，尤其是在数据处理时。












## 计算图

初步理解计算图可以看[这篇](https://zhuanlan.zhihu.com/p/191648279)

深入理解计算图还有一篇知乎上的，我点赞了，等水平上去了再去看

* 计算图是用来描述运算的**有向无环图**，有两个主要元素：节点 (Node) 和边 (Edge)。**节点表示数据**，如**向量、矩阵、张量**。**边表示运算**，如**加减乘除卷积**等。

* 理解 `torch.Tensor` 中与计算梯度相关的属性

  * `requires_grad`

    * PyTorch 计算图只追踪 `requires_grad=True` 的张量，而 Python 纯数值或 `requires_grad=False` 的张量在计算时**会被视为常数**，计算过程**不会对它们求导**。（ps：知乎上那张图也是，并没有将 `+1` 加入计算图中）

      * 比如在这段代码中，只有 `x` 和 `z` 在计算图中，4 和 `y` 均不在计算图中，但是这四个元素都进行了计算。
        ```python
        x = torch.tensor([3.0], requires_grad=True)
        y = torch.tensor([2.0], requires_grad=False)
        z = x * y * 4
        ```

        `z` 的计算过程可以等价为： $z = (x \times 2.0) \times 4$， 其中 `2.0` 和 `4` 都是常数，不会被追踪。如果调用 `z.backowrd()` ，查看 `x` 的梯度 `x.grad` 结果为 `8.0`，计算结果如下：$\frac{\partial_z}{\partial_x} = 2.0 \times 4 = 8.0$

  * `gard`

    * `tensor.grad` 存储张量的梯度信息。

      **`backward()` 计算梯度，非标量需提供 `gradient` 参数**。

      * ==针对向量情况要提供 `gradient` 参数==（后续有机会再去理解了）

      `grad` 会累积，需要 `zero_grad()` 清除。

      `detach()` 获取无梯度张量，`torch.no_grad()` 关闭梯度计算。

      `grad` 在优化器 (`optimizer.step()`) 中用于更新参数。

  * `grad_fn`

    * `grad_fn` 是 PyTorch 中张量（tensor）的一个属性，它表示**创建该张量的操作的计算图中的函数**，即记录了该张量是如何通过**前向传播**计算出来的。通过 `grad_fn`，PyTorch 能够自动追踪计算图，进行反向传播计算梯度。非叶子节点的 `grad_fn` 会指向相应的反向操作（如加法、乘法、矩阵乘法等），以便在反向传播时计算正确的梯度。
    * 叶子节点的 `grad_fn` 为 `None`，因为它们没有被其他操作生成。

  * `is_leaf`：其他所有节点都依赖于叶子节点。叶子节点的概念主要是为了节省内存，**在计算图中的一轮反向传播结束之后，非叶子节点的梯度是会被释放**的。

    * 自己思考，为什么非叶子节点的梯度会被释放？

      * 个人感觉，这一轮的参数更新只与这一轮的梯度有关。等到下一轮更新参数时，我们应该计算新的梯度来进行更新，计算累计梯度其实是不合理的。至于为什么叶子节点的梯度不会被释放？我不是很懂。

      * 首先需要明白，**叶子节点**（Leaf Tensors）：`requires_grad=True` 且**不是由计算得到**的张量（如模型参数）；**非叶子节点**（Non-Leaf Tensors）：由计算得到的张量，如 `z = x * y * 4` 这样的中间变量（**它们代表了一些中间过程**）。

        * 释放非叶子节点梯度的原因：（1）节省内存（2）通常不需要累积非叶子节点梯度，因为训练时，我们通常只关心**叶子节点**的梯度（比如模型参数），因为这些梯度用于优化更新。

          如果在反向传播结束之后仍然需要保留非叶子节点的梯度，可以对节点使用`retain_grad()`方法。
          ```python
          import torch
          
          x = torch.tensor([3.0], requires_grad=True)
          y = x * 2  # y 是非叶子节点
          
          y.retain_grad()  # 显式要求 PyTorch 保留 y 的梯度
          y.backward()
          
          print(y.grad)  # ✅ 现在可以访问，输出 1.0（因为 dy/dy = 1）
          print(x.grad)  # ✅ 输出 2.0（dy/dx = 2）
          
          ```

          

        * 为什么不释放**叶子节点**的梯度：（1）**梯度更新依赖它们**：叶子节点的梯度用于优化器（如 `Adam`、`SGD`）来更新参数。
          ```python
          optimizer.step()	# 使用参数的 .grad 来更新参数
          ```

          （2）**PyTorch ==默认==梯度是累积的**：PyTorch 设计上，叶子节点的 `.grad` 会累积梯度，而不是每次 `.backward()` 都覆盖之前的值。
          ```python
          optimizer.zero_grad()  # 需要手动清空梯度
          ```

          这样可以支持某些需要**梯度累积**的训练策略，比如梯度累积（gradient accumulation）。

          小批量梯度累计，多步反向传播。

          （3）最重要的一点是：事实上**在大多数训练情况下，我们都会使用 `optimizer.zero_grad()` 手动清空叶子节点的梯度**，而**梯度累积（gradient accumulation）属于相对少见的训练策略**。

* `.backward()` 这一过程到底在干麻？

  * 利用链式法则计算梯度，计算之后会丢弃计算图中非叶子节点的梯度，对计算图中的叶子节点，则会**累积梯度**，需要使用 `optimizer.zero_grad()` 来清空梯度。

