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

### Containers

#### [`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

* 所有神经网络模块的**基类**。我们模型也应该继承这个类。`Module` 还可以包含其他 `Module`，允许它们嵌套在树结构中。并在 `__init__` 方法中定义层结构，在 `forward` 方法中定义数据的前向传播过程。

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

  | 方法                 | 作用                                                     |
  | -------------------- | -------------------------------------------------------- |
  | `parameters()`       | 获取所有可训练参数                                       |
  | `named_parameters()` | 获取带名称的参数                                         |
  | `children()`         | 获取直接子模块                                           |
  | `named_children()`   | 获取直接子模块及名称                                     |
  | `modules()`          | 获取所有子模块（包括嵌套）                               |
  | `named_modules()`    | 获取所有子模块及名称                                     |
  | `state_dict()`       | 获取模型的状态字典                                       |
  | `load_state_dict()`  | 加载模型状态                                             |
  | `register_buffer()`  | **注册不参与优化的 `Tensor` （居然可以直接在外面实现）** |
  | `train(mode=True)`   | 设定训练模式                                             |
  | `eval()`             | 设定评估模式，会关闭 Dropout 及 BatchNorm 统计更新。     |
  | `apply(fn)`          | 对所有子模块应用函数（常用于初始化参数）                 |
  | `zero_grad()`        | 清空所有参数的梯度                                       |

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



## 计算图

初步理解计算图可以看[这篇](https://zhuanlan.zhihu.com/p/191648279)

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

