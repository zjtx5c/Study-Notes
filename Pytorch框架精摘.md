# pytorch 框架精摘

* 参考官方文档，知乎部分文献，李沐动手学深度学习v2

## `torch.nn`

目前暂定学习 `Containers`，`Linear Layers`， `Loss Functions`, `Transform_layers` 等模块...

### Buffer与 Parameter

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

### Containers

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

#### [`nn.Sequential`](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)

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



#### [`nn.ModuleList`](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)

* 在列表中保存子模块
* `nn.ModuleList` **可以像 Python 普通列表那样索引和遍历，其中的 `nn.Module` 子模块会被 PyTorch 正确注册**，因此它们可以被 `model.parameters()` 访问，并被优化器更新。
* 它的一些方法
  * `.append()`——追加一个 `nn.Module`（在末尾）
    * `layers.append(nn.Linear(20, 5))`

  * `.extend()`——扩展多个 `nn.Module`
    * `layers.extend([nn.Linear(5, 3), nn.Sigmoid()])`

  * `.insert()`——在指定索引插入 `nn.Module`
    * `layers.insert(1, nn.BatchNorm1d(20))  # 在索引 1 插入 BatchNorm 层`


#### [`nn.ModuleDict`](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html)

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

  - 权重 $W$ 的形状：`(out_features, in_features)`，因为数学上的写法是 $y = Wx + b$

  - 偏置 $b$ 的形状：`(out_features)`（可设置 true or false）

  - 输出 $y$ 的形状：`(*, out_features)`
    （`*` 表示任意额外的维度，如 batch_size）
  - **因此，我们的这个操作 `nn.Linear(in_feats, out_feats)` 定义的其实是 $W^{\top}$ 而非 $W$**，这是 PyTorch 设计中的一个关键细节，容易让人产生困惑。

* 关键细节

  * 权重初始化：**默认使用均匀分布** $\mathcal{U}(-\sqrt k, \sqrt k), k = 1 / in\_feates$，即我们**不需要显示初始化**
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

### [`init`](https://pytorch.org/docs/stable/nn.init.html)

权重初始化对深度学习模型的性能至关重要（比我们想象得重要），它决定了优化过程的起点。

具体原理可以看一下[李沐]()，以及我的《数学相关》笔记

这个模块中的所有函数都用于初始化神经网络的参数，因此它们都会在 `torch.no_grad()` 模式下运行（在此模式下，禁止计算图的构建，防止 `autograd` 记录计算历史。），并且**不会被自动求导（autograd）记录**（其实被记录了影响也不大，主要是**浪费计算资源**）。初始化阶段不涉及梯度计算，只是单纯地赋值。

* `torch.nn.init.calculate_gain`有什么用：**计算用于初始化神经网络权重的增益因子（gain factor）（根据非线性激活函数计算）**，以确保权重在前向传播和反向传播过程中不会导致梯度爆炸或消失。

  * 增益因子：是**用于缩放初始化权重的一个乘法系数**，它与神经网络中的激活函数有关，主要用于防止梯度消失或梯度爆炸。可以这样理解**增益因子是权重初始化方差的一个缩放系数**，但它本身不是方差。

    * **前向传播时**，输出的方差不要随着层数增加而急剧增大或减小。

      **反向传播时**，梯度的方差不会因为层数加深而爆炸或消失。

    * 在 Xavier 初始化中：$\text{Var}(\mathbf{W}) = \frac{2}{\text{feat}_{in}+\text{feat}_{out}} \cdot g^2$

* 具体细节技术可以参照[此篇](https://zhuanlan.zhihu.com/p/210137182)

  * 我们可以发现，若网络权重初始化随意，比如说直接初始化为标准正态分布 $\mathcal{N}(0,1)$ 会发现第一层的输出就为16左右，第二层200+，直接梯度爆炸（这是一个比较大的全连接层，每层有256个节点）。我们可以验证若网络比较小，可能选取 $\mathcal{N}(0, 0.01)$ 是有不错的数据稳定性效果的，但经过验证不能很好地处理过深的神经网络（例如100层）。
  * 直观感受梯度爆炸与合适初始化后每一层输出的 `std` 的平滑效果 

* 每种初始化方法都有它自己适用的场景，原则是保持每一层输出的方差不能太大，也不能太小。

### `Upsample`

上采样与插值







## [`torch.nn.functional`](https://pytorch.org/docs/stable/nn.functional.html)

`torch.nn.functional` 是 PyTorch 提供的一个函数式（functional）接口模块，它包含了许多用于构建和操作神经网络的函数。与 `torch.nn` 中的 `Module`（如 `torch.nn.ReLU`、`torch.nn.Linear`）不同，`torch.nn.functional` 里的函数是**无状态的**，这意味着它们不会自动保存参数，而是需要显式传入参数。

常见的有激活函数、归一化操作、损失函数、卷积操作、池化操作、线性变换等

* 需要理解什么时候使用 `torch.nn` ,什么时候使用 `torch.nn.functional` 以及这两者的差别

  * **推荐使用 `torch.nn`**

    - 需要自动管理参数的层，如 `nn.Linear`、`nn.Conv2d`、`nn.BatchNorm1d`。
    - 适用于构建神经网络模型时，参数会被 `model.parameters()` 管理。

    **推荐使用 `torch.nn.functional`**

    - 在 `forward` 方法中直接调用函数，如 `F.relu(x)`，避免额外创建 `Module` 实例。
    - 需要更灵活的计算，如手动控制 `dropout`，或在 `forward` 里直接使用 `F.conv2d()` 以避免 `nn.Conv2d` 额外存储权重。

### [Loss functions](https://pytorch.org/docs/stable/nn.functional.html#loss-functions)

* 损失函数？代价函数？目标函数？

#### `F.cross_entropy`

* 交叉熵的定义

  在信息论中，交叉熵衡量的是从一个真实分布 $ p $ 中获取一个事件的实际信息量与从预测分布 $ q $ 中获取相同事件的期望信息量之间的差异。给定一个真实分布 $ p(x) $ 和预测分布 $ q(x) $，交叉熵定义为：
  $$
  H(p, q) = -\sum_{x} p(x) \log q(x)
  $$
  

  其中：

  $ p(x) $ 是真实分布（即真实标签的概率分布）。

  $ q(x) $ 是预测分布（即模型预测的概率分布）。

  $\log$ 是对数函数，通常以自然对数为底。

  对公式可以这样子记忆：按照真实概率对现有的自信息 $- \log q(x)$ （即模型输出的混乱程度）进行加权累加，观察该模型输出的总体混乱程度。

* 用于**分类问题**，特别是多类分类。它计算了预测类别与真实类别之间的交叉熵。

  `loss = F.cross_entropy(input, target)`

  其中 `input` 是输出的 `logits` , `target` 则是 `ground_truth`，在此实现中，==**`logits`** **会自动通过 softmax 转换为概率，然后计算交叉熵损失。**==最终，`F.cross_entropy` 会默认返回整个批次的平均交叉熵损失。（若想返回每个批次的损失而不返回平均的损失，可以指定 `reduction = none`）















## `torch.Tensor`

直接看 [这篇](https://pytorch.zhangxiann.com/1-ji-ben-gai-nian/1.2-tensor-zhang-liang-jie-shao) 与 [这篇](https://pytorch.zhangxiann.com/1-ji-ben-gai-nian/1.3-zhang-liang-cao-zuo-yu-xian-xing-hui-gui)

### 概念

张量是包含单一数据类型元素的**多维矩阵**，它是标量、向量、矩阵的高维扩展。是 `pytorch` 中最基本的数据结构！

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

* 对 `dim` 的各种理解

  * 从压缩与聚合的角度理解：以 `torch.mean(x, dim = )` 操作为例

    `dim` 参数指定了 **要压缩的维度**，**即在哪个维度上进行聚合操作**（这里是求均值）。计算时会沿着指定维度进行压缩，其他维度保持不变。结果张量的形状是原始形状去掉被压缩的维度。

    记忆技巧，`dim = 0` 跨行计算，`dim = 1` 跨列计算

  * 针对高维中的 `dim` 如何理解？我现在已经完全悟了！==对哪一维操作，就是将这一维度的数据整体取出来，**以括号进行分隔**，分别对其求聚合（聚合操作时要对同一维度上的数据进行处理，位置要一一对应！）==，具体如下：

    我们以 `x = torch.arange(24).view(2, 3, 4)` 这个例子为例，它张这样

    ```python
    tensor([[[ 0,  1,  2,  3],
             [ 4,  5,  6,  7],
             [ 8,  9, 10, 11]],
    
            [[12, 13, 14, 15],
             [16, 17, 18, 19],
             [20, 21, 22, 23]]])
    ```

    当我们取 `dim = 0` 时，我们实际上是对这片数据进行处理：

    ```python
    [[...]],
    [[...]]
    ```

    其实就是

    ```python
    [[ 0,  1,  2,  3],
     [ 4,  5,  6,  7],
     [ 8,  9, 10, 11]]
    ,
     [[12, 13, 14, 15],
      [16, 17, 18, 19],
      [20, 21, 22, 23]]
    ```

    然后对这两组数据**对应位置**做要求的聚合操作即可！

    

    当我们取 `dim = 1`时，我们实际上是对这两片数据进行处理：

    ```python 
    [...],
    [...],
    [...]
    
    _____________________________
    
    [...],
    [...],
    [...]
    ```

    其实就是

    ```python
    [ 0,  1,  2,  3],
    _________________
    [ 4,  5,  6,  7],
    _________________
    [ 8,  9, 10, 11]
    
    _____________________________
    [12, 13, 14, 15],
    _________________
    [16, 17, 18, 19],
    _________________
    [20, 21, 22, 23]
    ```

    有两批数据，分别对上面这一批与下面这一批**对应位置上聚合**

    

    当我们取 `dim = 3` 时，我们实际上是对这么多批数据进行聚合操作（小的 `_____`  或者说逗号 `,` 用来区分同一维度上的数据，大的 `_____________________________` 用来“分批”）

    ```python
    0
    _____
    1
    _____
    2
    _____
    3
    _____________________________
    4
    _____
    5
    _____
    6
    _____
    7
    _____________________________
    .
    .
    .
    _____________________________
    20
    _____
    21
    _____
    22
    _____
    23
    
    ```

    

* 切分操作

  * `torch.chunk(chunks, dim)`

    **chunks**：这是要将张量分割成的子张量的数量。

    **dim**：指定要沿哪个维度进行分割，默认是 0。

    示例：

    ```python
    import torch
    
    # 创建一个张量
    x = torch.arange(8)
    
    # 将张量 x 沿 dim=0 维度分割成 4 个子张量
    chunks = x.chunk(4, dim=0)
    
    # 输出分割结果
    for chunk in chunks:
        print(chunk)
    ```

    输出：

    ```python
    tensor([0, 1])
    tensor([2, 3])
    tensor([4, 5])
    tensor([6, 7])
    
    ```

    1. 如果 `chunks` 不能整除张量的大小，**最后的几个子张量可能会比其他的子张量小**。
    2. `.chunk()` 方法不会改变原张量的形状，而是返回一个包含子张量的**元组**。（我们可以使用解包操作）

    这个操作通常用于**批处理数据**，或者将数据**分成小块进行并行处理**。（初见是在 LSTM 中的门并行处理）

* 索引操作

* 变换操作

  * `torch.stack()`	**这个在堆叠操作中是比较难理解的**（比较难想象）

    ==堆叠的时候按照维度将这些元素列出来，然后 `a` 与 `b` **对应的头或批次中的元素**一一堆叠，**用逗号隔开**，结束后用 `[]` **包裹升维**（表示堆叠结束），**不同维度之间也用逗号隔开**。==

    功能：用于沿着新的维度连接张量的函数（将一系列张量沿着一个新的轴进行堆叠）。它类似于 `torch.cat`，但是 `torch.stack` 会在指**定的维度上增加一个新的维度**（默认 `dim = 0`），而 `torch.cat` 是在现有的维度上拼接张量（默认也是沿着 `dim = 1` 进行拼接）。

    eg: 比如说下面分别是一个张量 `a`, `b`

    ```python
    tensor([[0, 1, 2],
            [3, 4, 5]]) 
     tensor([[ 7,  8,  9],
            [10, 11, 12]])
    ```

    我们使用 `torch.stack([a, b], dim = 0)` 这里 `dim = 0` 是**额外**在**最外面加一个维度**，可以这样感受一下 `[a, b]`，其中 `a` 与 `b` 之间用逗号隔开

    ==其实这个操作 `torch.stack([a, b], dim = 0)` 等价于 `torch.cat([a.unqueeze(0), b.unsqueeze(0)], dim = 0)`==

    最后变成这样：

    ```python
    tensor([[[ 0,  1,  2],
             [ 3,  4,  5]],
    
            [[ 7,  8,  9],
             [10, 11, 12]]])
    ```

    当然，`dim` 还可以指定其他维度，比如说 `dim = 1`，可以想象一下它张什么样子？有没有什么别的实际意义？和 `cat` 的区别是什么？

    下面均是 `torch.cat()` 操作，`dim` 分别为0, 1, 2

    ```python
    tensor([[[ 0,  1,  2],
             [ 3,  4,  5]],
    
            [[ 7,  8,  9],
             [10, 11, 12]]])
    tensor([[[ 0,  1,  2],
             [ 7,  8,  9]],
    
            [[ 3,  4,  5],
             [10, 11, 12]]])
    tensor([[[ 0,  7],
             [ 1,  8],
             [ 2,  9]],
    
            [[ 3, 10],
             [ 4, 11],
             [ 5, 12]]])
    ```

    

  * `torch.reshape()`
    功能：变换张量的形状。当张量**在内存中是连续**时，返回的张量和原来的张量共享数据内存，改变一个变量时，另一个变量也会被改变。

    * ==一定要注意==：在进行高维 `reshape` 的操作，一定要确保操作是安全的，比如在操作之前的张量相较于操作之后的张量要多一维或者少一维，其他维度保持一致，否则可能达不到预期的效果。

      **reshape 前后张量的元素数量必须一致，并且要考虑内存布局和维度的变化**。

      当我们希望张量的内存布局是连续的，或者需要用到 `view` 这种操作时，可以调用 `contiguous()` 来确保它是连续的。这能避免由于内存布局不连续导致的一些潜在错误或意外行为。

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

  * `torch.permute()`

    * `torch.permute(dims)`

      `dims` 是一个元组或列表，包含新的维度顺序。

      返回一个新的张量，维度顺序根据 `dims` 进行调整。

    * 可以和 `.view` 操作配合

  * **`torch.einsum()` 爱因斯坦求和约定** 

    `torch.einsum` 是 PyTorch 中用于执行爱因斯坦求和约定（Einstein summation convention）的一种函数，它提供了一种简洁且高效的方式来进行多维数组的操作。这个函数允许你指定张量操作的维度，并自动处理元素之间的求和和乘法。它可以用于执行矩阵乘法、点积、外积、转置等常见操作，并且在处理高维数据时非常方便。

    * 一般语法 `torch.einsum(equation, *operands)`

      * **equation**：一个字符串，表示你要执行的操作。这个字符串的格式与爱因斯坦求和约定类似，通过指定维度和操作符来定义。

        **operands**：输入的张量。

    * 例子

      > 1. 矩阵乘法  
      >
      > 矩阵乘法是最常见的应用之一，假设你有两个张量 $A$ 和 $B$，分别形状为 $(m, n)$ 和 $(n, p)$，它们的矩阵乘法结果是一个形状为 $(m, p)$ 的张量。使用 einsum 可以这样表示：  
      >
      > ```python
      > import torch
      > 
      > A = torch.randn(3, 4)  # 形状 (3, 4)
      > B = torch.randn(4, 2)  # 形状 (4, 2)
      > 
      > C = torch.einsum('ij,jk->ik', A, B)  # 结果形状 (3, 2)
      > ```
      >
      > `'ij,jk->ik'` 中，$i$, $j$, $k$ 是张量的维度，表示第一个张量 $A$ 的第一个维度与第二个张量 $B$ 的第一个维度进行匹配，最终计算出形状为 $(m, p)$ 的结果。  
      >
      > 
      >
      > 2. 点积  
      >
      > 如果你想计算两个向量的点积，假设 $x$ 和 $y$ 都是形状为 $(n,)$ 的向量，可以这样写：  
      >
      > ```python
      > x = torch.randn(4)
      > y = torch.randn(4)
      > 
      > dot_product = torch.einsum('i,i->', x, y)  # 输出标量
      > ```
      >
      > `'i,i->'` 表示对第一个和第二个张量的相同维度（即索引 $i$）进行乘法并求和，结果是一个标量。  
      >
      > 
      >
      > 3. 外积  
      >
      > 外积（outer product）也可以通过 einsum 实现，假设你有两个向量 $x$ 和 $y$，分别形状为 $(m,)$ 和 $(n,)$，它们的外积是一个形状为 $(m, n)$ 的矩阵：  
      >
      > ```python
      > x = torch.randn(3)
      > y = torch.randn(4)
      > 
      > outer_product = torch.einsum('i,j->ij', x, y)  # 输出形状 (3, 4)
      > ```
      >
      > `'i,j->ij'` 表示 $x$ 的第一个维度与 $y$ 的第一个维度进行外积操作。  
      >
      > 
      >
      > 4. 批量矩阵乘法  
      >
      > 假设你有两个三维张量 $A$ 和 $B$，它们分别形状为 $(b, m, n)$ 和 $(b, n, p)$，表示 $b$ 个矩阵的批量。你可以通过 einsum 进行批量矩阵乘法：  
      >
      > ```python
      > A = torch.randn(10, 3, 4)  # 批量大小 10，矩阵形状 (3, 4)
      > B = torch.randn(10, 4, 5)  # 批量大小 10，矩阵形状 (4, 5)
      > 
      > batch_matrix_product = torch.einsum('bmn,bnp->bmp', A, B)  # 输出形状 (10, 3, 5)
      > ```
      >
      > `'bmn,bnp->bmp'` 中的 $b$ 代表批量大小，$mn$ 和 $np$ 分别表示矩阵的维度，最终结果是形状为 $(b, m, p)$ 的张量。


### 张量的数学运算

* 常见的张量操作

1. **`torch.dot(a, b)`**:

   - 计算两个 **1D 张量** $a$ 和 $b$ 的点积（内积）。

   - 这两个张量必须有相同的大小。

   - **示例**：

     ```python
     a = torch.tensor([1, 2])
     b = torch.tensor([3, 4])
     result = torch.dot(a, b)  # 结果 = 1*3 + 2*4 = 11
     ```

2. **`torch.mul(a, b)`**: **会自动运行广播机制**

   - 执行张量 $a$ 和 $b$ 的逐元素乘法。如果两个张量形状相同，会逐元素相乘。

   - **示例**：

     ```python
     a = torch.tensor([1, 2])
     b = torch.tensor([3, 4])
     result = torch.mul(a, b)  # 结果 = [1*3, 2*4] = [3, 8]
     ```

3. **`torch.mv(mat, vec)`**:

   - 执行**矩阵-向量乘法**，其中 `mat` 是一个 **2D 张量（矩阵），`vec` 是一个 1D 张量（向量）**。矩阵的列数必须与向量的元素数量相同。

   - **示例**：

     ```python
     mat = torch.tensor([[1, 2], [3, 4]])
     vec = torch.tensor([5, 6])
     result = torch.mv(mat, vec)  # 结果 = [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
     ```

4. **`torch.matmul(a, b)`**: 会自动运用广播机制，从批次的角度理解高阶张量的处理（我的理解完全正确）

   - 执行矩阵乘法。如果 $a$ 和 $b$ 都是 2D 张量，它会执行常规的矩阵乘法。如果其中一个是 1D 张量，则会执行点积或向量-矩阵乘法，具体取决于维度。

   - 当然，还可以扩展到更高维度

   - **示例**：

     ```python
     a = torch.tensor([[1, 2], [3, 4]])
     b = torch.tensor([[5, 6], [7, 8]])
     result = torch.matmul(a, b)  # 结果 = [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]] = [[19, 22], [43, 50]]
     ```
     
     更==高维的操作==是要从批次的角度理解**（这里的批次必须一样或者有一项为1，使用广播机制）**
     
     ```python
     A = torch.randn(10, 2, 3)
     B = torch.randn(10, 3, 4)
     C = torch.matmul(A, B)
     # C.shape = (10, 2, 4)
     ```
     
     这里 A 中每个批次的形状是 `(2, 3)` B 中每个批次的形状是 `(3, 4)` ，可以进行矩阵乘法，最终得到一个 `(2, 4)` 的矩阵。
     
     **经过验证，若张量 A，B 的维度大于三维，那么最后两维看成矩阵，A 与 B 的维度必须与矩阵相乘的规矩匹配，前面的维度都可以看成批，A 与 B 的批要一一对应（这里考虑了广播机制）！！！**

5. **`torch.mm(a, b)`**:

   - 执行两个 2D 张量的矩阵乘法。这里的 `a` 和 `b` 都应该是二维矩阵（即形状是 (m,n)(m, n) 和 (n,p)(n, p)），矩阵乘法的结果是一个形状为 (m,p)(m, p) 的矩阵。

   - 需要注意的是，`torch.mm` 仅适用于二维张量（矩阵）。如果你想做更一般的矩阵与高维张量的乘法，可以使用 `torch.matmul`。（这个更加通用，多使用 `torch.matmul`）

   - **示例**：

     ```python
     a = torch.tensor([[1, 2], [3, 4]])  # 2x2 矩阵
     b = torch.tensor([[5, 6], [7, 8]])  # 2x2 矩阵
     result = torch.mm(a, b)  # 结果 = [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]] = [[19, 22], [43, 50]]
     ```



* 范数



* 张量的复制与维度扩展操作（**进行重复操作之前**建议使用 `.reshape()`  `.view()` `.unsqueeze()` 操作**扩展张量维度**） 

  * `torch.repeat()`（真正地**复制**数据，可以修改数据）

  * `torch.expand()` 与 `torch.expand_as()`（并没有真正地复制数据，不可以修改数据，只是运用了**广播机制**）

    > | 特性         | `repeat`                           | `expand`                           |
    > | ------------ | ---------------------------------- | ---------------------------------- |
    > | 操作方式     | 真实复制数据（内存变大）           | 创建 view，不复制数据（共享内存）  |
    > | 是否拷贝数据 | ✅ 是，数据会复制                   | ❌ 否，是广播机制                   |
    > | 是否占内存   | ✅ 是，变大                         | ❌ 否，节省内存                     |
    > | 是否可写     | ✅ 是                               | ❌ 否，不能写入（view 有限制）      |
    > | 使用场景     | 需要真正复制，例如输出给多个模型等 | 只读、广播用途，如注意力权重对齐等 |



### 对张量维度的深入理解与高阶操作（一般是以题目的形式给出）

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

* 如何使用 `F.cosine_similarity()` 函数计算一个二维 `tensor` （$n \times n$矩阵），两两之间的余弦相似度，输出应该仍为一个 $n \times n$ 矩阵。（涉及到扩张维度与张量的广播机制）

  要计算一个二维 `tensor`（大小为 $n \times n$）中两两之间的余弦相似度，可以通过以下步骤：

  1. **使用 `F.cosine_similarity` 计算每对行之间的余弦相似度**： `F.cosine_similarity` 计算的是两个张量之间的余弦相似度，通常用于一维向量之间。为了计算二维矩阵中每一对行之间的余弦相似度，我们可以通过矩阵的广播机制，将每一对行之间的余弦相似度计算出来。
  2. **通过广播技巧计算所有行的余弦相似度**：
     - 通过将二维张量与其转置（transpose）结合计算，可以得到所有行之间的余弦相似度矩阵。

  这里是实现的代码示例：

  ```python
  import torch.nn.functional as F
  
  # 假设我们有一个大小为 n x n 的矩阵
  n = 5  # 举个例子，5x5矩阵
  tensor = torch.rand(n, n)  # 随机生成一个 tensor
  
  # 计算两两行之间的余弦相似度
  cosine_sim_matrix = F.cosine_similarity(tensor.unsqueeze(1), tensor.unsqueeze(0), dim=2)
  
  print(cosine_sim_matrix)
  ```

  **解释：**

  - `tensor.unsqueeze(1)` 将原始的 `n x n` 矩阵扩展为大小为 $n \times 1 \times n$ 的张量，使得每一行变成一个单独的向量。
  - `tensor.unsqueeze(0)` 将原始的 `n x n` 矩阵扩展为大小为 $1 \times n \times n$ 的张量，使得每一列变成一个单独的向量。
  - `F.cosine_similarity` 的 `dim=2` 表示沿着第三个维度计算余弦相似度。

  最终，我们会得到一个大小为 $n \times n$ 的矩阵，每个元素表示原始矩阵中对应行之间的余弦相似度。

  * 自己的归纳总结：**我们要从==批次视角==以及==广播视角==理解这两个操作**。

    * 我们可以把**矩阵**视为一组向量，`unsqueeze` 操作的意义就在于将这些向量“包装”成一个批（batch）。通常在深度学习中，我们习惯将多个样本（比如向量）放在一个批量中处理。所以 `unsqueeze` 在某种程度上就是在告诉框架：“这些是独立的向量，接下来我要按批量进行操作。”

      `tensor.unsqueeze(0)` 就是将原始的 `n x n` 矩阵扩展为大小为 $1 \times n \times n$ 的张量。将其中的 $n \times n$ 当成一个批次来看待，第一维的 1 接下来会利用广播机制；`tensor.unsqueeze(1)` 可以认为现在有 $n$ 个批次，每个批次中有一个二维向量。

* 针对 `RGTN` 那个框架，如何使用**多头的思想**解决论文中的公式（7）~公式（10）

### 其他问题

* `torch.tensor` 和 `torch.Tensor` 的区别是什么？

  * `torch.tensor` 是用于创建张量的**函数**，并且会根据输入创建一个数据副本，它需要**显示**地提供数据`data`。
  * `torch.Tensor` 是**张量的类**构造函数，**通常**会返回一个未经初始化的张量（它其实是 `torch.FloatTensor` 的别名）。

  一般来说，建议使用 `torch.tensor` 来显式地创建张量，因为它的行为更加明确，尤其是在数据处理时。





## `torch.utils.data`

### `Dataset`

`Dataset` 是一个抽象类，用于表示**一个**数据集。你需要继承 `Dataset` 类并实现以下三个方法：

1. `__init__()`：初始化
2. `__len__()`：返回数据集的大小（即数据的总数量）。
3. `__getitem__()`：根据索引返回一个数据项（通常是输入数据和对应的标签）。

解决如何从磁盘中读取数据并把数据映射成 x, y 的形式。

* 为什么要继承 `Dataset` 这一个类（作用与优越性）

  继承 `Dataset` 类的目的是为了让我们的数据集能够与 PyTorch 的 **`DataLoader`** 一起使用，并且让数据集的访问过程更加标准化和便捷。具体来说，`Dataset` 是 PyTorch 中定义的数据集类，它帮助我实现了以下几点：

  1. 规范化数据集结构

  2. 与 `DataLoder` 一起工作

     `DataLoader` 是 PyTorch 中用于批量加载数据的工具，它会自动调用 `Dataset` 中的 `__getitem__()` 和 `__len__()` 方法来批量获取数据。因此，继承 `Dataset` 是为了让你的自定义数据集能够与 `DataLoader` 无缝配合工作。

     `DataLoader` 会：

     - 根据 `Dataset.__len__()` 方法知道数据集的大小，从而知道需要多少个批次（batch）。
     - 使用 `Dataset.__getitem__()` 方法来按需加载批次数据。

  3. 便于数据处理

     当我们继承 `Dataset` 时，我们可以自由地在 `__getitem__()` 方法中进行数据处理和变换。例如，我们可以在 `__getitem__()` 中对数据进行**归一化、标准化、裁剪、增强**等处理，从而将**数据预处理封装到数据加载的过程中**。



### `DataLoader`

`DataLoader` 是 PyTorch 中用于处理数据的一个非常重要的工具，它提供了一些功能来**简化**和**优化**模型的训练过程，特别是**当数据集较大或需要进行批量训练**时。

`DataLoader` 作为一个高效的==数据加载器==，可以**自动从 `Dataset` 类实例中提取数据，通过批量加载和打乱等操作，将数据送入模型进行训练。**

* 为什么要继承 `DataLoader` （其作用与优越性）

  1. 自动批量化（Batching）

     `DataLoader` 可以自动将数据集拆分成批次（batches），而且可以控制每个批次的大小。这使得在训练过程中，我们可以以较小的批次逐步喂入数据，从而避免一次性加载整个数据集带来的内存压力。

  2. 并行加载数据（Parallel Data Loading）

     `DataLoader` 可以通过多线程并行加载数据，这对大规模数据集尤其有用。它可以指定 `num_workers` 参数来使用多个工作进程（workers）加载数据，从而加速数据读取和预处理过程，避免了数据加载成为训练过程中的瓶颈。

  3. 打乱数据（Shuffling

     在训练过程中，为了避免模型过拟合并增强泛化能力，`DataLoader` 提供了 `shuffle=True` 的功能，它会在每个 epoch 结束后对数据进行重新打乱。这个功能对独立同分布（i.i.d.）数据尤为重要，然而对于时序数据，这个功能通常是关闭的。

  4. 自动处理数据加载的结束（Drop Last）

     当数据集的大小不能整除批次大小时，`DataLoader` 可以通过 `drop_last=True` 来丢弃最后一个小批次，避免小批次带来的不均衡影响。如果不想丢弃最后一个小批次，可以使用 `drop_last=False`（默认值）。

  5. 数据预处理与转换

     `DataLoader` 支持使用 `collate_fn` **自定义如何将数据样本聚合成一个批次**，可以应用各种预处理和转换。对于图像、文本、时间序列等不同类型的数据，可以定义特定的转换操作。

  6. 模块化设计与高效的内存管理与兼容性

     通过继承 `Dataset` 和使用 `DataLoader`，你可以确保你的数据加载逻辑与 PyTorch 的训练框架兼容，特别是在**分布式训练、GPU 加速**等场景中，`DataLoader` 提供了与 PyTorch **深度集成的优化**。
  
* 如何理解 `DataLoader` 的批处理机制，在具有时序特点的数据集上该如何处理.

  * 是否打乱样本（`shuffle`） 决定了 `DataLoader` 是否顺序加载数据
  * 若样本不是批次的整数倍，那么若 `drop_last=False` ，即便最后一批不足 `batch_size`，也会被保留。
  * 综上，我们可以不打乱样本以及实现对时序数据的批处理







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

## 自动求导机制

* 标量的链式法则我们很熟悉，那么拓展到向量呢？
  * 最核心的问题还是要把形状搞对？
  
* 为什么需要自动求导？

  * 因为神经网络动不动就几百层，如果我们手写求导是几乎不现实的，因此需要自动求导

* 自动求导计算一个函数在**指定值上**的导数，它有别于**符号求导**与**数值求导**

* 理解自动求导的两种模式

  * 正向累积，从前向后求导数：`forward`（DL特别耗GPU资源的罪魁祸首，因为需要存储所有中间变量用于反向传播，等反向传播结束后再释放中间变量的内存）
  * 反向累积，又称反向传递 `backward`：从后往前求导数
  * 正向求导和反向累积本质上没什么区别，无非就是正向求导不存数，我只存一个数，不停更新这个数，最后得到导数，而反向累积就是把数都存起来，到时候求什么就取需要的数来用

* ==注意一下**非标量变量的反向传播**==

  * 当`y`不是标量时，向量`y`关于向量`x`的导数的最自然解释是一个矩阵。 对于高阶和高维的`y`和`x`，求导的结果可以是一个高阶张量。

    然而，虽然这些更奇特的对象确实出现在高级机器学习中（包括深度学习中）， 但当调用向量的反向计算时，我们通常会**试图计算一批训练样本中每个组成部分的损失函数的导数**。 这里，我们的目的不是计算微分矩阵，而是单独计算批量中**每个样本的偏导数之和**。即，一般来讲我们对向量都是求和让其成为一个标量（因为损失通常都是一个标量）再进行反向传播

    eg:

    `y.sum().backward()`

    `等价于y.backward(torch.ones(len(x)))`

* 分离计算

  有时，我们希望将某些计算移动到记录的计算图之外。 例如，假设`y`是作为`x`的函数计算的，而`z`则是作为`y`和`x`的函数计算的。 想象一下，我们想计算`z`关于`x`的梯度，但由于某种原因，希望将`y`视为一个常数， 并且只考虑到`x`在`y`被计算后发挥的作用。

  这里可以分离`y`来返回一个新变量`u`，该变量与`y`具有相同的值， 但丢弃计算图中如何计算`y`的任何信息。 换句话说，梯度不会向后流经`u`到`x`。 因此，下面的反向传播函数计算`z=u*x`关于`x`的偏导数，同时将`u`作为常数处理， 而不是`z=x*x*x`关于`x`的偏导数。

  若我们以后有需求将网络固定，那么 `detach` 是一个很有用的功能

* Python控制流的梯度计算

  使用自动微分的一个好处是： 即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度。 在下面的代码中，`while`循环的迭代次数和`if`语句的结果都取决于输入`a`的值。

  ```python
  def f(a):
      b = a * 2
      while b.norm() < 1000:
          b = b * 2
      if b.sum() > 0:
          c = b
      else:
          c = 100 * b
      return c
  ```


## 梯度裁剪

* **强烈推荐**看这[一篇](https://zhuanlan.zhihu.com/p/99953668)通过感受其例子来加深理解

  ==自己写代码去验证一下== （get）

梯度裁剪（Gradient Clipping）是一种防止梯度爆炸的优化技术，它可以在**反向传播过程中**对梯度进行**缩放或截断**，使其保持在一个合理的范围内。梯度裁剪有两种常见的方法：

（1）按照梯度的绝对值进行裁剪，即如果梯度的绝对值超过了一个阈值，就将其设置为该阈值的符号乘以该阈值。
（2）按照梯度的范数进行裁剪，即如果梯度的范数超过了一个阈值，就将其按比例缩小，使其范数等于该阈值。例如，如果阈值为1，那么梯度的范数就是1。
在PyTorch中，可以使用 `torch.nn.utils.clip_grad_value_(model.parameters(), clip_value = 0.5)` 和 `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10, norm_type = 2)` 这两个函数来实现梯度裁剪，它们都是在梯度计算完成后，更新权重之前调用的。

* 防止梯度太大导致步子跨得太大而错过损失函数最小值

* 工作流程
  ```
  outputs = model(data)：前向传播，计算模型的输出。
  loss = loss_fn(outputs, target)：计算损失函数。
  optimizer.zero_grad()：清零所有参数的梯度缓存。
  loss.backward()：反向传播，计算当前梯度。
  nn,utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)：对梯度进行裁剪，防止梯度爆炸。
  optimizer.step()：更新模型的参数。
  ```

* 何时需要梯度裁剪

* （1）深度神经网络：深度神经网络，特别是RNN，在训练过程中容易出现梯度爆炸的问题。这是因为在反向传播过程中，梯度会随着层数的增加而指数级增大。

  （2）训练不稳定：如果你在训练过程中观察到模型的损失突然变得非常大或者变为NaN，这可能是梯度爆炸导致的。在这种情况下，使用梯度裁剪可以帮助稳定训练。

  （3）长序列训练：在处理长序列数据（如机器翻译或语音识别）时，由于序列长度的增加，梯度可能会在反向传播过程中累加并导致爆炸。梯度裁剪可以防止这种情况发生。

  需要注意的是，虽然梯度裁剪可以帮助防止梯度爆炸，但它不能解决梯度消失的问题。对于梯度消失问题，可能需要使用其他技术，如门控循环单元（GRU）或长短期记忆（LSTM）网络，或者使用残差连接等方法。

* 源码如下
  ```python
  def clip_grad_value_(parameters, clip_value):
      r"""Clips gradient of an iterable of parameters at specified value.
      Gradients are modified in-place.
  
      Arguments:
          parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
              single Tensor that will have gradients normalized
          clip_value (float or int): maximum allowed value of the gradients.
              The gradients are clipped in the range
      """
  
      if isinstance(parameters, torch.Tensor):
          parameters = [parameters]
      clip_value = float(clip_value)
      for p in filter(lambda p: p.grad is not None, parameters):
          p.grad.data.clamp_(min=-clip_value, max=clip_value)
  ```

  



## 采样技术

### `torch.multinomial`

`torch.multinomial` 是 PyTorch 中用于从给定的一维概率分布中 **无放回（或有放回）地随机采样** 的函数，常用于需要按概率采样索引的任务中，比如负采样、采样子集等。

* 函数原型

```python
torch.multinomial(input, num_samples, replacement = False, *, generator = None, out = None) -> Tensor
```

* 参数说明：

**input**：一维或二维的张量，表示每个元素的概率权重，必须是非负的（可以为 0），不需要归一化。

**num_samples**：要采样的样本数量。

**replacement**：是否**有放回**采样。默认为 `False`，表示**无放回**。

**generator**：可选，用于控制随机数生成器。

**out**：可选，输出张量。

* 返回值：

返回一个长整型张量（`LongTensor`），表示所采样元素的索引。

如果所有的权重（`weights`）都相等，或者说每个元素的值都是 1，那么就相当于对每个样本进行 **等概率的随机采样**，即每个样本被采到的概率相等。

* eg：

```python
weights = torch.tensor([1, 1, 1, 1, 1])
samples = torch.multinomial(weights, 3, replacement=False)
print(samples)
```

* **针对批量采样（一对多），我们有放回地采样（要有批的思想）**

可以参考这个写法（**是用于计算 LTR 损失的采样过程**）

```python
def list_loss(y_pred, y_true, list_num=10, eps=1e-10):
    '''
    y_pred: [n_node, 1]
    y_true: [n_node, 1]
    '''
    n_node = y_pred.shape[0]

    ran_num = list_num - 1
    indices = torch.multinomial(torch.ones(n_node), n_node*ran_num, replacement=True).to(y_pred.device)

    list_pred = torch.index_select(y_pred, 0, indices).reshape(n_node, ran_num)
    list_true = torch.index_select(y_true, 0, indices).reshape(n_node, ran_num)

    list_pred = torch.cat([y_pred, list_pred], -1) # [n_node, list_num]
    list_true = torch.cat([y_true, list_true], -1) # [n_node, list_num]

    list_pred = F.softmax(list_pred, -1)
    list_true = F.softmax(list_true, -1)

    list_pred = list_pred + eps
    log_pred = torch.log(list_pred)

    return torch.mean(-torch.sum(list_true * log_pred, dim=1))
```



### `torch.index_select`

`torch.index_select` 是 PyTorch 中的一个函数，用于从输入张量中**根据指定的索引选取指定维度上的元素**。它允许你通过给定一个索引数组来选择输入张量中相应位置的元素。

* 函数原型

```python
torch.index_select(input, dim, index) -> Tensor
```

* 参数说明：

**input**：输入的张量。

**dim**：沿着哪个维度进行索引操作（例如，对于二维张量，`dim=0` 是选择行，`dim=1` 是选择列）。

**index**：一个长整型张量，表示你要选择的索引。该张量的形状**必须是 1D**，且其中的元素是要选择的索引。

* 返回值

返回一个新的张量，包含了根据给定索引选择的元素。

* eg: 我们从二维张量中按行索引选择
