# Frame

## 一些基本且重要但是之前不太注意到的操作

### 针对异构图 `g` ，如何遍历图的每条边并获取边的 `(src, rel, dst)` 索引信息

* 1. **获取所有边的规范类型**

  - 使用 `g.canonical_etypes` 获取图中所有边的规范类型，形式为 `(src_type, rel_type, dst_type)`。

  2. **遍历每种边类型**

  * 对于每种边类型，使用 **`g.edges(etype=(src_type, rel_type, dst_type))`** 获取该类型边的源节点和目标节点索引。

* 尝试练习，遍历每条边，获得每条边的三元组形式 `idx_s relation idx_o`

  * 会用到 `zip`

### 添加自环操作

`dgl.add_self_loop(g)`是用来在图中添加自环（self-loops）的方法。自环指的是节点与其自身之间的边。这个方法将会在图 `g` 中的每个节点上添加一条指向自身的边。这个图对象**可以是异构图（heterograph）或者同构图（homograph）**。
新增的边会在原始的 `edges` 的后面，并不会打乱之前的索引顺序

* 自环的应用

  * **图卷积**：在图神经网络中，通常需要考虑节点与其邻居（包括自身）的关系。添加自环是确保每个节点在计算其表示时能包含自身信息的常见操作。

    **图分析**：自环有时也用于某些图的算法中，比如页面排名算法（PageRank）等。

  * 检查图中是否有自环，使用 `g.has_self_loop()`

    `True`：表示图中包含至少一个自环。

    `False`：表示图中不包含自环。



## dgl.NID

`dgl.NID` 是 DGL 中的一个**特殊保留键（reserved key）**，用于表示：**一个节点在原始图（parent graph）中的 ID**。

* 为什么我们需要 `dgl.NID`？

  在图神经网络的训练中，我们经常会从原图中：

  1. 采样出一个子图
  2. 构造一个 block （二部图）
  3. 在 block 中进行消息传递（message passing）

  但是，这时我面对的是**子图**或**block**的“局部图结构”，里面的节点编号是**局部编号**，**和原图不一致**！所以，为了能知道这些“局部节点”在原图中是谁，DGL 用 `dgl.NID` 来存储“它在原图中的编号”。例如（从 xxxdata 中获取，即配合 `.xxxdata` 这个方法）：

  | 表达式                   | 作用                                              |
  | ------------------------ | ------------------------------------------------- |
  | `block.srcdata[dgl.NID]` | 得到 block 中 src 节点在原图中的 ID（int Tensor） |
  | `block.dstdata[dgl.NID]` | dst 节点在原图中的 ID                             |
  | `block.edata[dgl.EID]`   | block 中的边对应原图中的边 ID                     |

  







## dgl.DGLGraph

### `g.apply_edges()`

* `g.apply_edges(func, edges='__ALL__', etype = None)`

* 自己的理解：主要就是用于更新边的特征，默认情况下，遍历图中的所有边，**根据源节点 (u) 与目标节点 (v) 的特征**计算新的**边特征**，需要将计算结果存储到 `graph.edata` 中。

* 使用 `.apply_edges(edge_attention)` 会为图的边数据（`edata`）**添加一项新的数据** `g.edata['e']`（若返回的是 `{'e': xxx}`）。

  > ### **📌 `DGLGraph.apply_edges` API 解析**
  >
  > `DGLGraph.apply_edges(func, edges='__ALL__', etype=None)` 用于 **更新指定边的特征**，并存储到 `graph.edata` 中。
  >
  > ------
  >
  > ## **🔹 1. 函数作用**
  >
  > 该函数的作用是：
  >
  > 1. **遍历图中的某些边（或所有边）**
  > 2. **基于源节点（u）和目标节点（v）的特征计算新的边特征**（当然也可以直接修改边权）
  > 3. **将计算结果存入 `graph.edata`**
  >
  > ------
  >
  > ## **🔹 2. 参数解析**
  >
  > ### **① `func`（必填）：边更新函数**
  >
  > **该参数用于定义如何计算边特征**，可以是：
  >
  > - [DGL 内置函数（`dgl.function`）](https://www.dgl.ai/dgl_docs/api/python/dgl.function.html#api-built-in)
  >
  >   - `fn.u_add_v('x', 'y', 'z')`：计算 $z = x_u + y_v$
  >   - `fn.u_mul_v('x', 'y', 'z')`：计算 $z = x_u \times y_v$
  >   - `fn.copy_u('x', 'z')`：将**源节点** `x_u` 复制给边 `z`
  >
  > - 自定义函数
  >
  >   - 传入 `lambda` 或 `def` 函数，自由定义边特征计算方式。
  >
  > - `func`的返回值：
  >
  >   - `func` 需要返回 **一个字典**，格式为：
  >
  >     ```python
  >     {'边特征名1': 计算值1, '边特征名2': 计算值2, ...}
  >     ```
  >
  >   * 计算值的 **形状必须与边的数量匹配**，否则会报错。
  >
  > ### **② `edges`（可选，默认 `'__ALL__'`）**
  >
  > **选择要更新特征的边**，可以有以下格式：
  >
  > | **类型**                         | **解释**                                           |
  > | -------------------------------- | -------------------------------------------------- |
  > | `int`                            | 仅更新 **一条边**（ID）                            |
  > | `Int Tensor`                     | **多个边 ID** 的张量                               |
  > | `iterable[int]`                  | **多个边 ID** 的 Python 可迭代对象（列表等）       |
  > | `(Tensor, Tensor)`               | **节点对格式**，两个张量分别存储源节点和目标节点   |
  > | `(iterable[int], iterable[int])` | **Python 可迭代的节点对**，类似 `(Tensor, Tensor)` |
  > | `默认值 '__ALL__'`               | **所有边**                                         |
  >
  > ### **③ `etype`（可选）**
  >
  > **用于处理异构图（Heterogeneous Graph）**，可以是：
  >
  > - `(str, str, str)` 格式
  >   - `(src_type, edge_type, dst_type)`，分别表示 **源节点类型、边类型、目标节点类型**。
  > - 单个字符串
  >   - 当图中**每种边类型唯一**时，可以直接写 `edge_type`。
  >
  > ------
  >
  > ## **🔹 3. 代码示例**
  >
  > ### **📝 示例 1：使用 `fn.u_add_v`**
  >
  > ```python
  > import dgl
  > import torch
  > import dgl.function as fn
  > 
  > # 创建一个图（3条边）
  > graph = dgl.graph(([0, 1, 2], [1, 2, 3]))  # 0->1, 1->2, 2->3
  > 
  > # 给每个节点加一个特征
  > graph.ndata['h'] = torch.tensor([[1.], [2.], [3.], [4.]])  # 4 个节点，每个节点 1 维特征
  > 
  > # 计算边特征 e = u.h + v.h
  > graph.apply_edges(fn.u_add_v('h', 'h', 'e'))
  > 
  > # 查看边特征
  > print(graph.edata['e'])  
  > ```
  >
  > **输出**
  >
  > ```
  > tensor([[3.],  # 1 + 2  (0->1)
  >         [5.],  # 2 + 3  (1->2)
  >         [7.]]) # 3 + 4  (2->3)
  > ```
  >
  > ------
  >
  > ### **📝 示例 2：自定义 `apply_edges`**
  >
  > ```python
  > def edge_update(edges):
  >     return {'e': edges.src['h'] * edges.dst['h']}  # e = u.h * v.h
  > 
  > graph.apply_edges(edge_update)  # 计算 e = h_u * h_v
  > print(graph.edata['e'])
  > ```
  >
  > **输出**
  >
  > ```
  > tensor([[2.],  # 1 * 2  (0->1)
  >         [6.],  # 2 * 3  (1->2)
  >         [12.]]) # 3 * 4  (2->3)
  > ```
  >
  > ------
  >
  > ### **📝 示例 3：仅更新部分边**
  >
  > ```python
  > # 仅更新第 1 条边（索引为 0 的边 0->1）
  > graph.apply_edges(fn.u_add_v('h', 'h', 'e'), edges=0)
  > print(graph.edata['e'])  # 仅第 1 条边被更新
  > ```
  >
  > ------
  >
  > ## **🔹 4. 总结**
  >
  > - **`apply_edges` 用于计算边特征**，基于 **源节点（u）和目标节点（v）** 的特征进行计算。
  > - **支持 DGL 内置函数 `fn.u_add_v` / `fn.u_mul_v`，也支持自定义 `lambda` 函数**。
  > - **可选择更新部分边（`edges` 参数）**，默认为所有边。
  > - **支持异构图（`etype` 参数）**，用于多种节点类型的图。
  >
  > ------
  >
  > ### **✅ 总结一句话**
  >
  > 🚀 **`apply_edges(func)` = 遍历所有（或指定）边，基于 `u` 和 `v` 计算 `e`（大部分情况下），然后存入 `graph.edata`**。



### `g.update_all()`

* `message_func`: 消息函数，定义如何在边上计算消息。
  
  * 默认传入**边**（`edges`），可以直接使用 `edges.src` 与 `edges.dst`
  * `message_func` 计算得到的消息会**自动存入目标节点 （即入度节点）的 `mailbox`**，然后 `reduce_func` 从 `nodes.mailbox` 里取出这些消息进行聚合。
    * 并不会直接存入 `g.ndata/g.nodes.data` 中
  
* `reduce_func`：聚合函数，定义如何在节点上聚合接收到的消息。
  
  * 默认传入**点**（`nodes`），注意对 `nodes.mailbox()` 的理解
  * 从 `nodes.mailbox` 里取出这些消息进行聚合，然后**计算新特征，并存入 `g.ndata[h_new]`**
  * 该步骤结束后，**`mailbox` 会被清除**，因为 DGL 认为它已经被用完了，如果在此之后尝试访问`mailbox` 则会出现 `NoneType` 访问错误（**血泪教训**）。
  
* `apply_node_func`（可选）：用于对聚合后的节点特征进行额外的处理。

  * 输入 `nodes`

  * 可以辅助打印一些信息

* ==注意==：在 DGL 的 `update_all` 机制中，`message_func` 计算得到的消息会传递给目标节点，并存储在 `nodes.mailbox` 中。`nodes.mailbox['e']` 之所以会比 `message_func` 计算出的消息==多出一个维度==，是因为 DGL 需要存储**一个节点从多个源节点接收的所有消息**。

  * 在 `message_func`（如 `fn.copy_u('h', 'm')`）计算出的消息形状通常是 `(E, d)`，其中每条边都有一个对应的消息。

  * **在 `reduce_func` 运行之前，`nodes.mailbox['e']` 的形状是：** $(B, K, d)$

    * **$B$**：当前 batch 处理的目标节点数（即 `update_all` 作用的目标节点数）。

      **$K$**：每个目标节点的入度（即它接收的消息数，取决于入边数）。

      **$d$**：单条边上的消息维度。



## [dgl.function](https://www.dgl.ai/dgl_docs/api/python/dgl.function.html)

* 一般诸如 `fn.u_add_v()` 这种内联函数**在 `apply_edges()` 和 `update_all()` 中的行为不同**，
  - **在 `g.apply_edges()` 中使用时**，计算结果会存入 **`edata`**（边数据字典）。
  - **在 `g.update_all()` 中使用时**
    - 针对 `message_func`： 计算结果会先存入 **`edata`**（临时存储），然后被传递到 **`mailbox`**，用于目标节点的聚合。
    - 针对 `reduce_func`（比如使用 `fn.sum('m', 'ft')`）: 会将上一轮任务得到的 `mailbox` 提取数据并将最终的结果存入到目标节点 `ndata` 中。

### [`fn.u_add_v`](https://www.dgl.ai/dgl_docs/generated/dgl.function.u_add_v.html)

* 这是一个内嵌函数，所以通常要配合 `g.apply_edges(fn.u_add_v(...))` 使用

* 操作机制

  * `fn.u_add_v('el', 'er', 'e')`会对**每条边**的源节点特征('el')和目的节点特征('er')进行相加

  * 这个操作是**逐边(edge-wise)**执行的，而不是在全局张量上操作

  * 它看起来是针对点的操作（其实看起来也不是），但其实是**针对边的操作**，eg，当形状不匹配时如

    * `el.shape = [2,3,1]`，`er.shape = [4,3,1]`的第一个维度不同

      但DGL会根据边的实际**连接关系**，为每条边选取对应的源节点和目的节点特征进行相加

      最终输出的'e'特征的形状会是`[num_edges, 3, 1]`

  * 在 `g.apply_edges` 的配合下，最终将特征 `'e'`  存储到 `g.edata['e']` 中

### [`fn.u_mul_e`](https://www.dgl.ai/dgl_docs/generated/dgl.function.u_mul_e.html)

* 操作机制
  * `fn.u_mul_e` 计算 **源节点特征 (`u`)** 和 **边特征 (`e`)** 的逐元素乘积，并将结果作为消息传递到目标节点。

### [`fn.v_dot_u`](https://www.dgl.ai/dgl_docs/generated/dgl.function.v_dot_u.html)

* 操作机制

  * `fn.v_mul_u` 计算 **汇点特征 (`v`)** 和 **源点特征 (`u`)** 的点积，在最后一维计算（顺序不能乱）。

* `g.apply_edges(fn.v_dot_u('dst_k', 'src_k', 't'))`  的等价实现是：

  ```python
  def edge_func(edges):
      src_k = edges.src['src_k']
      dst_k = edges.dst['dst_k']
      # 因为特征的维度有时候不是一维的，可能会带有头，所以我们不使用 `torch.dot` 而是使用 `torch.mul` 逐元素相乘再求和的方式来计算每个维度上的点积
      t = torch.mul(src_k, dst_k).sum(dim = -1, keepdim = True)
      return {'t': t}
  g.apply_edges(edge_func)
  ```

  特别要注意的是这里的 $v$ 要与 `dst_k` ，$u$ 要与 `src_k` 一一对应起来。
  
  此外若特征为2维 `[H, D]`，那么执行 `fn.v_dot_u` 或者 `fn.u_dot_v` 的操作后，得到的 `t` 的 `shape` 为
  
  `[E, H, 1]`


## dgl.nn

### 异构图学习模块（重点学习）

#### [卷积 `HeteroGraphConv`](https://www.dgl.ai/dgl_docs/generated/dgl.nn.pytorch.HeteroGraphConv.html#dgl.nn.pytorch.HeteroGraphConv)

这是一个通用的卷积模块，用于在异构图上执行卷积操作。异构图中的不同类型的节点和边需要特定的卷积操作，这个模块允许用户自定义这些卷积方式，以便在不同类型的节点和边上进行处理。

* 理解聚合到底是针对什么聚合（场景需要清楚），理解其 `forward`函数在什么时候触发？又具体做了什么？输入有两种形式，输入节点的特征，可以是字典（`dict[str, Tensor]`）形式，或者在某些情况下为元组（`tuple`），如果是元组，它将被视为源节点和目标节点特征的对。

* 理解下源码（知道流程）==一定要通透==

  * 复现一下这个代码，==一定要通透！==
  
  * `init(self, mods, aggregate = "sum")` 在干麻？
    
    * 将异构图不同的边类型与对应的卷积模块这一字典封装到 `nn.ModuleDict`中：`self.mods = nn.ModuleDict(mods)`（`mods` 有两种传入方式：（1）三元组（2）直接谓词传入）
    * 之后再处理一下聚合模块。
    
  * `forward(self, g, inputs, mod_args = None, mod_kwargs = None)`在干麻？
    
    * （一）两种方法传入`inputs`字典（1）直接传入节点类型及其对应的二维`Tensor`（数量，特征），此时 `inputs` 是一个字典（2）传入一个 `tuple` ，分别代表源点 `src` 和 汇点的 `dst`。（此时 `inputs` 是一个元组，里面存放着两个字典）
    * 比较关键（1）首先遍历规范化类型边，并通过 `rel_graph = g[stype, etype, dtype]` **获取子图**，若传入的 `src` 属于该子图的一部分则（2）调用 `etype` 对应的卷积模块在该子图 `rel_graph` 上进行对应的图卷积操作，**注意**：我们传入是通过 `tuple`即`(inputs[stype], outputs[stype])` 传入具体的图卷积中（如`GAT`）（3）得到边类型为 `etype` 的情况下**汇点 `dst` 的特征**（4）最终通过 `init` 过程中传入的 `aggregate` 对不同边类型 `etype` 下得到的特征进行聚合！
    
  * 一些细节
  
    * `str(src_type, rel_type, dst_type)` 这样写是不支持的。这不是 `str()` 的合法用法，因此 Python 会抛出 `TypeError`。但是我们可以鸡贼地这样处理一下 `tmp  = (src_type, rel_type, dst_type)` 然后再 `str(tmp)` 就可以处理了~（当然源码并不是这样写得）
    * 在 `init` 过程中
      * `nn.ModuleDict` 并不支持 `get()` 方法 （`dict.get()` 方法是 Python 字典 (`dict`) 的一个内置方法，用于根据键获取对应的值，**并允许在键不存在时返回默认值，而不会引发 `KeyError`。** 例如：`dict.get(key, default = None)`），因此，为了防止使用 `nn.ModuleDict` 去查找不存在的边引发 `keyError`, 源码中维护了两个 `dict`。`self.mod_dict` 直接存了原始的 `mods`，用于 `get()` 方法查询（避免 `ModuleDict` 没有 `get()`）
      * 有一个解决 0 入度节点机制
      * 定义聚合方式，貌似是一个黑盒
  
    * 在 `forward` 过程中
      * 记得处理对应的模型参数 `mod_args` 与 `mod_kwargs` 
    
    * ==为什么要使用子图`rel_graph`传入卷积模块？==你可能会认为是为了节省计算资源，但核心原因并不是这个
      * 为了节省计算资源
      * 通过 `rel_graph = g[stype, etype, dtype]`  处理的图其实只有一种边类型的，那么它在模块中就不是异构图而是同构图了！这在我们自定义编写卷积模块是非常方便，比如当我们在模块内部写 `graph.apply_edges(fn.u_add_v('el', 'er', 'e'))` 就不需要特地针对异构图指定  `etype` 了
* ==自定义卷积模块需要注意的点（重点！！）==
  
  * 一定需要明确的是，若我们使用 `dgl.nn.HeteroGraphConv` 来管理不同的边类型的图卷积，那么进入图卷积的特征输入 `h` 一定是一个 `tuple` 类型，这是 `dgl.nn.HeteroGraphConv` 在 `forward` 过程中自动帮我们处理好的，`tuple` 中分别装得是 `src_feat` 和 `dst_feat` 的 `tensor`！也就是说在若我们在使用 `dgl.nn.HeterGraphConv` 的同时若要自定义卷积模块，那么我们必须将输入以 `tuple` 的形式设置。


#### [线性变换 `HeteroLinear`](https://www.dgl.ai/dgl_docs/generated/dgl.nn.pytorch.HeteroLinear.html)

#### [嵌入 HeteroEmbedding](https://www.dgl.ai/dgl_docs/generated/dgl.nn.pytorch.HeteroEmbedding.html)

这是一个为异构图创建嵌入表的模块。在异构图中，不同类型的节点或边通常会有不同的嵌入，`HeteroEmbedding` 可以根据节点的类型为每个节点生成对应的嵌入向量。

* 直接看官网更方便

* 输入和输出都与**字典**相关（感觉异构图多多少少都和字典有关）

* 尝试自己写一下源码，其生成的原理是基于 `nn.Embedding`（大概懂了）

* 理解一下 `nn.Embedding`

  > * **初始化**：词汇表大小，表示有多少个不同的单词或类别；每个单词或类别对应的嵌入向量维度
  > * **输入**：`nn.Embedding` 的输入是一个整数张量，通常表示单词的**索引**。例如，句子中的每个单词可以映射为一个唯一的整数。
  > * **输出**：`nn.Embedding` 的输出是一个浮点数张量，表示每个索引对应的向量。这些向量通常是**可学习的参数**。

##### 注意点

* `nn.ModuleDict()` **只能接受字符串作为键**，自己写源码时不要忘记添加。

#### [`TypedLinear`](https://www.dgl.ai/dgl_docs/generated/dgl.nn.pytorch.TypedLinear.html#typedlinear)

 



## dgl.ops

在 DGL 中，`dgl.ops` 和 `dgl.function` 都提供了用于图操作的功能，但它们有一些关键的区别：

* **`dgl.ops`**：提供的是一些底层的、常用的图操作函数，通常用于对图**进行直接的操作或计算**。例如，`edge_softmax`、`scatter_add`、`segment_sum` 等。这些操作大多是图相关的数值操作，或者是一些高效的聚合、归一化和计算功能。通常，它们用于图神经网络（GNN）的实现中，执行消息传递、特征聚合等操作。

  **`dgl.function`**：提供的是在消息传递机制中更高层次的操作功能，**允许用户自定义消息传递函数的行为**。`dgl.function` 允许你定义消息的计算、聚合的方式以及节点或边的更新方式。通常，`dgl.function` **提供的是框架内部的功能封装，更多地用于定义图神经网络的模型结构**。

### [`edge_softmax()`](https://www.dgl.ai/dgl_docs/generated/dgl.nn.functional.edge_softmax.html#dgl.nn.functional.edge_softmax)

$$
a_{ij}=\frac{\exp (z_{ij})}{\sum_{k \in \mathcal{N(i)}} \exp(z_{ik})}
$$

默认情况下，边 softmax 由**目标节点规范化**

*  貌似 `from dgl.nn.functional import edge_softmax` 和 `from dgl.ops import edge_softmax` 这两份 `edge_softmax` 的**源码都是一样的**
* 直接看官网的例子，照着敲一敲，需要注意以下几点
  * 去看源码
  * 理解参数 `norm_by`
  * 理解维度
* 异构图上的边归一化

注意该函数的输入要求。DGL 的 `edge_softmax` 接收形状为 `[num_edges, num_heads, 1]` 或 `[num_edges, 1]` 的张量（最后一维得是标量，因为要做的是归一化操作）：

- 它会在 **以每个目标节点为单位** 的边集合中，对该维度上的注意力值做 softmax。

- 通常在多头 attention 中，我们的注意力是 `[num_edges, num_heads]`，每个 head 一个值，但 `edge_softmax` 需要最后一维是 `1`（标量）用于 softmax 操作。

  因此，若想要通过 `t = g.edata.pop("attn")` 的方式取出来，需要关注下其形状：`[E, H, 1]` 或者 `E, 1` ，我们可以使用 `.sum(dim = -1)` 的操作将其变成 `[E, H]` 或者 `[E,]`



## 批处理与采样相关技术

首先，可以参考一下[这篇文档](https://blog.csdn.net/beilizhang/article/details/112966162)进行入门

* 事实上，**批处理（batching）和 block 构建**是一项技术，**图采样（sampling）**是另一项技术，它们本质上是相互独立的。然而，如果将这两者结合使用，可以进一步降低显存占用，从而支持大规模图的训练。尽管如此，这种内存优化通常会带来一定程度的性能下降（例如表示质量下降或训练误差增大），这是一个常见的精度与效率的权衡问题。事实上，如果我们使用**批处理（batching）+ 全邻居采样（full neighbor sampling）**，**理论上精度几乎不会改变**，因为我们保留了图的完整拓扑结构信息。只是每次训练时只处理部分目标节点（batch），但每个目标节点的邻居信息是完整的。但是若是我们使用 transformer 模型，那么精度可能会有**较大的误差**，虽然图的拓扑结构没有改变，但是注意力矩阵从全图变成了子图。

* 思考子图 transfrom 和全图 transfrom 的差异与区别

  | 模型                            | 优点                                                         | 缺点                                                         |
  | ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | ✅ Block/Batch-based Transformer | 低计算成本；可处理大图；可以做动态采样；训练时 memory-friendly | **视野有限**，每个节点只能看到其局部邻居；**跨子图信息**难以建模 |
  | ❌ 全图 Transformer              | **表达力强**，可建模长距离依赖；无需采样，结构完整           | **计算与内存开销极高**；难以扩展到百万规模图                 |

* 思考以下邻居采样的工作原理

  我们在更新一个节点的表示时，**只需要该结点以及其邻居**在**上一层中的表示**。所以需要找出节点消息传递的依赖。

* 边界子图（Frontier）与 block 有什么区别与联系？

* **如果我们打算使用 block 更新全图，那么 dstnode 也应该是图中的所有节点，既然这样那么使用 block 相较于直接更新全图的优势在哪？**（即我们需要理解为什么使用 block + 采样 + 批处理技术能够有效缓解显存压力）

  * 个人感觉是分批次节点采样，降低空间复杂度

* `dgl.sampling.sample_neighbors(g, seed_nodes, fanout)` 和 `dgl.in_subgraph(g, seed_nodes)` 有什么区别？

  前者是**按固定数量随机采样邻居**（**稀疏采样**）。后者是**不做采样，保留所有入边邻居（全邻居采样）。**

### `dgl.in_subgraph()`

我们将**有依赖关系的图**称为边界子图（frontier）。DGL有多种生成边界子图的函数，这里使用最常用的 `dgl.in_subgraph()`，它的参数为**原始图**和**指定节点（汇点）**，它**根据指定节点的入边**生成边界子图。它用于从原图中提取给定节点的“入子图”（in-subgraph），即保留所有**指向这些节点的边**，常用于处理节点的**入邻居信息**。例如：

```python
frontier = dgl.in_subgraph(g, [8])
```

注意：

* 若是同构图，那么该子图的 type 类型与原图的 type 类型是一致的。
* 该函数返回的子图中会将不需要的边去掉，**节点仍然保留**，也即节点总数与原图是不变的

### `dgl.to_block()`

`dgl.to_block` 是 DGL 中将一个 **子图（subgraph）** 转换为一个 **block 图** 的函数，用于支持 GNN 中的 **message passing** 机制（源节点 → 目标节点结构）。**DGL 中的 `block` 图，本质上就是一个“定向的二部图（directed bipartite graph）”。**下面是它的详细参数说明。

注意理解这里的**明确构造二部图的含义**：即明确地划分了 `srcdata` 与 `dstdata`

* 函数参数

  ```python
  dgl.to_block(g, dst_nodes=None, include_dst_in_src=True, copy_ndata=True, copy_edata=True)
  ```

  | 参数名               | 类型                                  | 默认值 | 说明                                                         |
  | -------------------- | ------------------------------------- | ------ | ------------------------------------------------------------ |
  | `g`                  | `DGLGraph`                            | 必需   | 输入**子图**（可以是方向图、多边图等），会被转换为一个 block（bipartite）结构 |
  | `dst_nodes`          | Tensor[int] or dict[str, Tensor[int]] | `None` | 指定要作为 block 右边（目标）的节点 ID如果为 `None`，默认使用 `g.dstnodes()`。**我们可以通过手动选定目标节点来更新我们希望更新的节点的表示。**但一般情况下我们不会仅更新个别节点。 |
  | `include_dst_in_src` | `bool`                                | `True` | 控制是否将目标节点 `dst_nodes` 也包含进 `src_nodes` 中（即左边输入端也包含目标节点）**也就是默认会自动构建自环，这样就能保证孤立节点也能进行消息传递。** |
  | `copy_ndata`         | `bool`                                | `True` | 是否从原图复制节点特征到 block 图中                          |
  | `copy_edata`         | `bool`                                | `True` | 是否从原图复制边特征到 block 图中                            |

* 返回值

  返回一个新的  `DGLBlock` 类型图对象（子类于 `DGLGraph`）

  有以下典型结构：

  ```python
  block.srcdata[dgl.NID]  # 原图中源节点ID
  block.dstdata[dgl.NID]  # 原图中目标节点ID
  block.edges()           # 方向为：src → dst
  ```

* 说明

  在 DGL 的 block 中：

  - 一侧是 **源节点（src nodes）** → 输入信息（提供特征）
  - 一侧是 **目标节点（dst nodes）** → 聚合邻居信息（更新特征）
  - 二部图中节点的特征来源？模型输入时，**src 节点需要提供初始特征**，用于聚合给 dst 节点

* 为什么 GNN 中使用二部图（block）

  | 目的                                    | block 的优势                                                 |
  | --------------------------------------- | ------------------------------------------------------------ |
  | **消息传递清晰**                        | 只从 src 向 dst 传播，避免冗余计算                           |
  | **利于批处理训练**                      | 每个 batch 只更新 dst 节点，**src 是上下文（需要显式提供）** |
  | **减少计算冗余**                        | 不用计算未采样的邻居节点的输出表示                           |
  | **适配 Transformer/GAT 风格 Attention** | 明确区分查询（dst）和键值（src）                             |

* 操作方式

  block 的操作方式与常规二分图差不多（其实也和 `dgl.graph` 差不多）

  * `.number_of_src_nodes()` 和  `.number_of_dst_nodes()` 获取源节点和目标节点的个数。

  * `.srcnodes()` 和 `.dstnodes()`

    返回**独立于原图的重新编号**的源点和汇点的**编号**。

  * `.srcdata["h"]` 和 `.dstdata["h"]` **访问**输入节点和输出节点特征。当然也可以自己赋值。

    需要注意的是，block会对边界子图中的节点和边重新编号，可以通过 dgl.NID 得到块中输入节点和输出节点的初始节点ID，可以通过 dgl.EID 得到边ID到输入边界中边的初始ID的映射，且DGL确保块的输出节点将始终出现在输入节点中。

### 多层 GNN 中的依赖关系

这里引出了一个比较重要的概念就是说，在多层 GNN 中**样顺序其实是从“输出层往输入层”倒推的**。这是为了逐层反推出 GNN 每一层所需的输入邻居节点。

* 图神经网络 `massage passing` 的实际流程是：

  我们要更新某个节点 `u` 的表示（例如输出层表示），就必须先拿到它邻居的表示：

  - 所以我们要知道**谁是它的一阶邻居**（用于最后一层）；
  - 再去找这些邻居的邻居（用于倒数第二层）；
  - 一层层地往前推（其实就是倒推）。

* 举一个例子进行说明

  假设我们有一个 2 层的 GNN 模型，结构如下

  ```css
  Input → Layer 1 → Layer 2 → Output
  ```

  加入现在想更新某一批目标节点的表示（比如训练样本中的节点），这个时候：

  | block_id | 对应 GNN 层 | 是为了计算哪一层的输入？ | 要采哪些邻居？       |
  | -------- | ----------- | ------------------------ | -------------------- |
  | 0        | Layer 2     | 计算输出层（最终表示）   | 采目标节点的一阶邻居 |
  | 1        | Layer 1     | 计算中间表示             | 采邻居的邻居         |

  因此，采样顺序是从 block 0 开始，往回采样，从“模型最后一层”需要的输入开始，一层层向输入层**回溯**。

  可以理解成要进行**两次遍历**：**第一遍采样子图（反向），第二遍前向传播（正向）**。这种设计是 DGL Block 模式中非常经典且高效的做法，特别适用于大图训练。

### 异构图上的采样

搁置

### 自定义邻居采样器

先看之前常用的 `MultiLayerFullNeighborSampler` 采样器的实现。该采样器的实现如下：

```python
class MultiLayerFullNeighborSampler(dgl.dataloading.BlockSampler):
    def __init__(self, n_layers):
        super().__init__(n_layers)

    def sample_frontier(self, block_id, g, seed_nodes):
        frontier = dgl.in_subgraph(g, seed_nodes)
        return frontier
```

其中：`super().__init__(n_layers)` 是告诉父类，我要采样多少层。这个必写！

容易看出，采样器继承自 `dgl.dataloading.BlockSampler` ，而 `BlockSampler` 负责调用 `sample_frontier()` 函数生成块列表，所以主要任务还是落在了 `sample_frontier()` ，它需要从原始图中根据指定节点生成边界子图。



再来看更复杂的邻居采样器 `MultiLayerNeighborSampler`的实现：

```python
class MultiLayerNeighborSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts):
        super().__init__(len(fanouts))

        self.fanouts = fanouts

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id]
        if fanout is None:
            frontier = dgl.in_subgraph(g, seed_nodes)
        else:
            frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)
        return frontier
```

与 `MultiLayerFullNeighborSampler` 不同的是，它会根据 `fanout` 值随机从指定节点邻居中选出固定数量的邻居。具体体现在`frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)` 。其中

1. `block_id`: 当前正在采样的 GNN 层的编号（从外层到内层，比如第0层表示输出层对应的输入）。

2. `g`: 当前的图。

3. `seed_nodes`: 本层的目标节点（我们要更新这些节点的表示）。

于是，只需稍加修改 `sample_frontier()` 函数，我们就可以**自定义一个邻居采样器**。下面实现的是**以某种概率**将种子节点的入边随机剔除。

```python
class MultiLayerDropoutSampler(dgl.dataloading.BlockSampler):
    def __init__(self, p, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.p = p

    def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):
        # 获取种 `seed_nodes` 的所有入边
        src, dst = dgl.in_subgraph(g, seed_nodes).all_edges()
        # 以概率p随机选择边
        mask = torch.zeros_like(src).bernoulli_(self.p)
        src = src[mask]
        dst = dst[mask]
        # 返回一个与初始图有相同节点的边界
        frontier = dgl.graph((src, dst), num_nodes=g.number_of_nodes())
        return frontier

    def __len__(self):
        return self.n_layers
```

**注意，边界子图需要保持节点与原始图节点相同。**

### 异构图上自定义采样器

搁置



### NodeDataLoader

`dgl.dataloading.NodeDataLoader` 用于构建一个基于 DGL 的多层图神经网络训练的数据加载器（NodeDataLoader），其中**每一批数据**都是通过**邻居采样**方式从图中获取的。

* 主要参数（一般只关注前三个）

  ```python
  dgl.dataloading.NodeDataLoader(
      g,
      nodes,
      sampler,
      *,
      use_ddp=False,
      device=None,
      batch_size=1,
      shuffle=False,
      drop_last=False,
      num_workers=0,
      collate_fn=None,
      worker_init_fn=None,
      use_uva=False,
      pin_memory=False,
      persistent_workers=False,
      **kwargs
  )
  ```

  事实上，如果我们使用全邻居采样 + 批处理 block 技术，进行训练，~~理论上是没有精度误差的~~。我们选择一个例子进行说明：

  ```python
  # full sampler，定义全邻居采样器
  sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
  
  dataloader = dgl.dataloading.NodeDataLoader(
      g,                    # 原始图
      train_idx,            # 要训练的目标节点 ID（种子）
      sampler,              # 使用的采样器
      batch_size=args.batch_size,	# 这里默认是用了2048个
      shuffle=True,         # 打乱数据顺序
      drop_last=False,      # 最后一批不足 batch_size 是否丢弃
      pin_memory=True,      # 提升 CPU -> GPU 拷贝效率
      num_workers=args.num_workers  # 多进程采样加速
  )
  ```

  这个 `NodeDataLoader` 会：

  - 每次从 `train_idx` 中采出一个批次（batch）种子节点；
  - 用 `sampler` 从图中按层采样邻居，构造一组 mini-batch 的**Block 子图**；
  - **返回这组 Block 子图给模型做==前向传播==**。
  - `shuffle = True` 是为了让每次采样出来的子图（block）不一样。
  
* 遍历过程与内部拆解

  我们一般会这样遍历这个图数据装载器

  ```python
  for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader)
  ```

  `input_nodes`：

  * 表示 **当前 batch 所需的所有输入节点 ID**（即 src nodes），它们可能会被用来聚合邻居信息（通过多层 GNN）。

  * 包括：

    - 当前要更新的目标节点的邻居（来自上一 hop 的节点）；

    - 当前目标节点本身；

  * 换句话说，它是前向传播中每一层所需要的**所有节点的集合**（我们对其不更新，它只充当上下文的作用）。

  `output_nodes`：

  * 表示 **当前 batch 中真正要更新/计算表示的目标节点 ID**（即 dst nodes）；

  * 在训练时，只有这些节点的表示会用于计算 loss；

  * 它是采样器传入的种子节点（seed nodes），也叫作 mini-batch 中的中心节点。

  `block`：

  * 是一个长度为 `num_layers` 的列表，表示从原图中采样得到的 **多层 mini-batch 边界子图**，每一项是一个 `Block`。
  * 每一层 `Block` 包含：
    - 当前 GNN 层所需的邻接信息；
    - `src` 表示输入节点集合，`dst` 表示目标节点集合；
    - 注意：这些是有向图中每一层 GNN 的计算图。

### 大致训练流程

1. 设置采样器，一般以**全邻居采样器**为主

   ```python
   sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
   ```

2. 设置数据装载器（要好好**理解这个数据装载器的构成**！）我们一般考虑将其打乱， `drop_last = False`，并设置为多线程，根据自己的显存压力大小设置 `batch_size` 

   注意我们还需要计算多少步 `step` 才能训练完一轮 `epoch`

   ```python
   dataloader = dgl.dataloading.NodeDataLoader(
       g,
       train_idx,
       sampler,
       batch_size=args.batch_size,
       shuffle=True,
       drop_last=False,
       # pin_memory=True,
       num_workers=args.num_workers
   )
   
   total_step = len(train_idx) // args.batch_size + 1
   ```

3. 开始训练，理解这里的所有数据结构

   ```python
   for epoch in range(args.epochs):
       
       for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
           # load the input features and output labels
           blocks = [block.int().to(device) for block in blocks]
           batch_struct = blocks[0].srcdata['struct_feat']
           batch_content = blocks[0].srcdata['content_feat']
           batch_labels = blocks[-1].dstdata['labels']
   
           # forward
           model.train()
           batch_pred, loss = model(blocks, batch_struct, batch_content, batch_labels)
   
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           # .... # 
   ```

4. 需要自己重写模型批处理的逻辑。

   先搁置了，以后有时间好好研究





# Models

## GAT

### 理解并熟悉GAT注意力分数 $e_{ij}= \text{LeakyReLU}(\alpha \top [Wh_{i} || Wh_{j}])$ 的计算流程

#### 逐元素相乘 + 求和 与 拼接 + 线性变换是等价的

> **1️⃣ 原始 GAT 公式**
>
> 在 GAT 论文中，注意力分数 $e_{ij}$ 计算如下：
> $$
> e_{ij} = \text{LeakyReLU} \left( \mathbf{a}^T [W\mathbf{h}_i || W\mathbf{h}_j] \right)
> $$
>
> 其中：
> - $W$ 是 **(out\_feats, in\_feats)** 形状的线性变换矩阵。
> - $\mathbf{h}_i \in \mathbb{R}^{in\_feats}$ 是节点 $i$ 的输入特征。
> - $[\cdot || \cdot]$ 表示 **拼接**（concatenation）。
> - $\mathbf{a} \in \mathbb{R}^{2 \cdot out\_feats}$ 是 **可学习的注意力参数**。
>
> 展开后：
>
> $$
> e_{ij} = \text{LeakyReLU} \left( \sum_k a_k (W \mathbf{h}_i)_k + \sum_k a_{k + out\_feats} (W \mathbf{h}_j)_k \right)
> $$
>
> 其中 $k$ 遍历 $out\_feats$ 维度。
>
> **2️⃣ 逐元素相乘+求和的变形**
>
> 现在看 GAT 代码的实现：
>
> ```python
> el = (feat * self.attn_l).sum(dim=-1, keepdim=True)  # (N, num_heads, 1)
> er = (feat * self.attn_r).sum(dim=-1, keepdim=True)  # (N, num_heads, 1)
> graph.apply_edges(fn.u_add_v('el', 'er', 'e'))  # (E, num_heads, 1)
> ```
>
> 这里的 `self.attn_l` 和 `self.attn_r` 是 **两组不同的参数**，形状是 `(num_heads, out_feats)`，分别用于 **源节点** 和 **目标节点**。
>
> 等式展开：
>
> $$
> \text{el}_i = \sum_k (W \mathbf{h}_i)_k \cdot a_{l,k}, \quad
> \text{er}_j = \sum_k (W \mathbf{h}_j)_k \cdot a_{r,k}
> $$
>
> 然后 `u_add_v` 计算：
>
> $$
> e_{ij} = \text{LeakyReLU} \left( \text{el}_i + \text{er}_j \right)
> $$
>
> **3️⃣ 等价性证明**
>
> 我们对比：
> 1. **原始 GAT 计算**：
>
> $$
> e_{ij} = \text{LeakyReLU} \left( \sum_k a_k (W \mathbf{h}_i)_k + \sum_k a_{k + out\_feats} (W \mathbf{h}_j)_k \right)
> $$
>
> 2. **代码中的计算**：
>
> $$
> e_{ij} = \text{LeakyReLU} \left( \sum_k (W \mathbf{h}_i)_k \cdot a_{l,k} + \sum_k (W \mathbf{h}_j)_k \cdot a_{r,k} \right)
> $$
>
> 这两者等价的关键在于：
> - **在原始 GAT 中，拼接后使用的 $\mathbf{a}$ 向量可以拆成两部分：**  
>   - **$\mathbf{a}_{l} = (a_1, ..., a_{out\_feats})$**  
>   - **$\mathbf{a}_{r} = (a_{out\_feats+1}, ..., a_{2\cdot out\_feats})$**  
> - **代码中的 `attn_l` 和 `attn_r` 本质上就是 $\mathbf{a}_l$ 和 $\mathbf{a}_r$**，所以数学上是等价的。
>
> **4️⃣ 结论**
> $$
> \mathbf{a}^T [W\mathbf{h}_i || W\mathbf{h}_j] = \sum_k a_{l,k} (W \mathbf{h}_i)_k + \sum_k a_{r,k} (W \mathbf{h}_j)_k
> $$
>
> 即 **逐元素相乘+求和的形式** 与 **拼接+矩阵乘法** 是等价的！
>
> **为什么这样实现更好？**
>
> 1. **减少计算量**：
>    - 拼接后要计算 $2 \times out\_feats$ 维的点积，而逐元素相乘只需要计算 $out\_feats$ 维的点积，两者本质上计算相同的内容，但减少了内存访问的开销。
>
> 2. **更适合张量计算**：
>    - `feat * attn_l` 直接用 **广播** 机制，避免了额外的维度扩展（拼接后的 `a^T` 需要额外的 `matmul` 操作）。
>
> 所以，这种实现方式不仅数学上等价，而且计算更快，适合 GPU 加速。
>

* ==自己尝试证明（repeat）==
  * 自己写了一个结论：$A_{n \times m} \times b_{m \times 1} = A^{(l)}_{n \times m_l} \times b ^{(u)}_{m_l \times1} + A^{(r)}_{n \times m_r} \times b^{(d)} _{m_r \times 1}$
    * 其中 $A = cat([A^{l},A^{r}], dim = 1)$    $b = cat([b^{u}, b^{d}], dim = 0)$
  * 个人感觉 **逐元素相乘+求和的形式** 与 **拼接+矩阵乘法** 是等价的原因是 ==**矩阵相乘的中间维度的扩展**==！，**沿着中间维度求和**
    * eg：$A_{a \times b} \times B_{b \times c}$ ==**可以在 $b$ 的那一维上扩展**==
    * $A$ 看行向量， $B$ 看列向量！

* 思考并彻底理解加入多头机制后相关的运作流程（个人感觉本质上是要理解**广播机制**）
  * 自己写了下草稿貌似懂了

#### 从广播机制的视角理解多头注意力的处理

* ==利用广播（broadcasting）+ 逐元素相乘（element-wise multiplication）+ 求和（sum）可以在高维度上实现矩阵乘法！==（这种操作可以实现==多头注意力、批量计算等场景==，可以避免手写 `for` 循环，提高计算效率）

  * 广播机制的条件：如果两个张量的形状（shape）不同，它们可以进行逐元素运算的条件是
    1. **从右向左对齐维度**（即**后面的维度必须匹配**）。
    2. **某个维度不匹配时**，其中一个维度必须是 `1`，否则无法广播。
  * **广播机制（broadcasting）** 设计的主要目的是 **快速批量处理数据，减少显式的循环，提高计算效率**，特别是在 **深度学习、数值计算和矩阵运算** 中，广播机制起到了关键作用。
  * 自己的脑子要清楚，里面到底发生了什么？（横的变竖的）
  
  > ### **🔹 1. 传统矩阵乘法**
  >
  > 假设有两个矩阵：
  >
  > - $ A \in \mathbb{R}^{m \times n} $
  > - $ B \in \mathbb{R}^{n \times p} $
  >
  > 矩阵乘法 $ C = A B $ 计算方式如下：
  >
  > $C_{i,j}= \sum_{k=1}^{n}A_{i,k}B_{k,j}$
  >
  > 其中：
  >
  > - $ A $ 的每一行与 $ B $ 的每一列进行**点积**。
  >
  > ### **🔹 2. 如何用广播机制实现？**
  >
  > 如果想用 **逐元素相乘 + 求和** 来实现矩阵乘法，我们可以扩展维度，使得两个矩阵在高维上 **广播**：
  >
  > 1. 扩展 A 的最后一维
  >    - $ A \in \mathbb{R}^{m \times n} $
  >    - 变形为 **$ A \in \mathbb{R}^{m \times n \times 1} $**
  > 2. 扩展 B 的前一维
  >    - $ B \in \mathbb{R}^{n \times p} $
  >    - 变形为 **$ B \in \mathbb{R}^{1 \times n \times p} $**
  > 3. 利用广播机制进行逐元素相乘
  >    - $ A' * B' \in \mathbb{R}^{m \times n \times p} $
  > 4. 沿着 `n` 维度求和
  >    - $ \sum_{k=1}^{n} A' B' $ → 得到 $ C \in \mathbb{R}^{m \times p} $
  >
  > ### **🔹 3. 代码示例**
  >
  > ```python
  > import torch
  > 
  > A = torch.randn(3, 4)  # (m, n)
  > B = torch.randn(4, 5)  # (n, p)
  > 
  > # 使用广播机制 + 逐元素相乘 + 求和 实现矩阵乘法
  > C = (A.unsqueeze(-1) * B.unsqueeze(0)).sum(dim=1)  # (m, n, p) -> sum over n -> (m, p)
  > 
  > # 对比 PyTorch 原生矩阵乘法
  > C_builtin = A @ B  # (m, p)
  > 
  > print(torch.allclose(C, C_builtin))  # True，说明两种方法等价
  > ```
  >
  > ### **🔹 4. 为什么这样可以工作？**
  >
  > 1. ```python
  >    A.unsqueeze(-1)
  >    ```
  >
  >    - 形状变为 `(m, n, 1)`，模拟 `A` 在 `n` 维度上的广播。
  >
  > 2. ```python
  >    B.unsqueeze(0)
  >    ```
  >
  >    - 形状变为 `(1, n, p)`，模拟 `B` 在 `m` 维度上的广播。
  >
  > 3. ```pyhon
  >    A * B
  >    ```
  >
  >    - **利用广播机制**，直接得到 `(m, n, p)`，相当于对所有 `m` 和 `p` 的组合，计算 `n` 个元素的乘积。
  >
  > 4. ```python
  >    sum(dim=1)
  >    ```
  >
  >    - 沿 `n` 维度求和，实现矩阵乘法的最终结果 `(m, p)`。
  >
  > ------
  >
  > ### **🔹 5. 适用于更高维度的矩阵**
  >
  > 如果 `A` 和 `B` 具有更高的维度（比如 batch 处理），可以利用同样的方法：
  >
  > ```python
  > A = torch.randn(10, 3, 4)  # (batch, m, n)
  > B = torch.randn(10, 4, 5)  # (batch, n, p)
  > 
  > C = (A.unsqueeze(-1) * B.unsqueeze(-3)).sum(dim=-2)  # (batch, m, n, p) -> sum over n -> (batch, m, p)
  > 
  > C_builtin = A @ B  # (batch, m, p)
  > 
  > print(torch.allclose(C, C_builtin))  # True
  > ```
  >
  > ------
  >
  > ### **🔹 6. 总结**
  >
  >  ✅ **广播机制** 允许在 **高维空间** 中进行矩阵乘法，而不需要 `for` 循环。
  >  ✅ 通过 **扩展维度 + 逐元素相乘 + 求和**，我们可以实现 **高效的矩阵乘法**，特别适用于 **GPU 加速**。
  >  ✅ 这种方式适用于 **多头注意力、批量计算等场景**，可以避免手写 `for` 循环，提高计算效率。
  >
  >  ✅ 不是真的复制数据，而是“假装”数据扩展了（只扩1）
  >
  >  ✅ 利用维度扩展 + 逐元素计算 + 计算后求和

#### 深刻理解高维度张量的样子与形状（这章在 `Pytorch` 中进行详细学习）

* `unsqueeze(-1)` 是在最里面套一层“括号”; `unsqueeze(0)`是在最外面套一层“括号”

#### 在代码实现上处理归一化的细节（手写）

* 针对 `GAT` 的归一化我们一般使用自带的函数 `alpha = dgl.nn.functional.edge_softmax(g, e)` 来进行处理，其中 `g` 为一个DGL图，`e` 为 Tensor，表示图中边的权重。表示计算每个节点的所有出边的Softmax。

* 其实是对公式细节的理解没有到位：
  $$
  \alpha^{(l)}_{i,j} = \frac{\exp(e^{(l)}_{i,j})}{\sum_{k \in \mathcal{N}(i)} \exp(e^{(l)}_{i,k})}
  $$

这里的归一化意思是：选定一个**入点 / 出点**$i$   ，对其相邻的入边/出边做归一化（**以点为中心**）；而不是无脑对所有边做归一化！这两者的意义完全不一样。**已经经过本人的实验验证**。

* 手写时还需要想清楚 `g.nodes.mailbox['e']` 中的维度情况。
* 此公式对应流程中的消息聚合模块



### 理解每个过程中的维度与形状到底是怎么样的

* ==建议自己调试一遍并且自己实现一遍，融汇贯通==

* 形状的理解（配合 `dgl.nn.HeteroGraphConv` 使用的情况下）：

  * 输入 `(src_feat_dim, dst_feat_dim), dst_out_feat_dim, num_heads`


  * 输出的维度 $(N', \text{num of heads}, feat\_out)$（未经 `aggreagte` 处理的情况下）
    $N'$ 表示这一轮（一般对应一个子图）的 `dst` 的数量；$feat \_ out$ 表示输出的特征维度



