# Frame

## 一些基本且重要但是之前不太注意到的操作

### 针对异构图 `g` ，如何遍历图的每条边并获取边的 `(src, rel, dst)` 索引信息

* 1. **获取所有边的规范类型**

  - 使用 `g.canonical_etypes` 获取图中所有边的规范类型，形式为 `(src_type, rel_type, dst_type)`。

  2. **遍历每种边类型**

  * 对于每种边类型，使用 **`g.edges(etype=(src_type, rel_type, dst_type))`** 获取该类型边的源节点和目标节点索引。

* 尝试练习，遍历每条边，获得每条边的三元组形式 `idx_s relation idx_o`

  * 会用到 `zip`


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


## dgl.nn

### 异构图学习模块

#### 卷积

#### [嵌入 HeteroEmbedding](https://www.dgl.ai/dgl_docs/generated/dgl.nn.pytorch.HeteroEmbedding.html)

* 直接看官网更方便

* 输入和输出都与**字典**相关（感觉异构图多多少少都和字典有关）

* 尝试自己写一下源码，其生成的原理是基于 `nn.Embedding`（大概懂了）

* 理解一下 `nn.Embedding`

  > * **初始化**：词汇表大小，表示有多少个不同的单词或类别；每个单词或类别对应的嵌入向量维度
  > * **输入**：`nn.Embedding` 的输入是一个整数张量，通常表示单词的**索引**。例如，句子中的每个单词可以映射为一个唯一的整数。
  > * **输出**：`nn.Embedding` 的输出是一个浮点数张量，表示每个索引对应的向量。这些向量通常是**可学习的参数**。

##### 注意点

* `nn.ModuleDict()` **只能接受字符串作为键**，自己写源码时不要忘记添加。



 

# Models

## GAT

### 理解并熟悉GAT注意力分数 $e_{ij}=LeakyReLU(\alpha \top [Wh_{i} || Wh_{j}])$ 的计算流程

#### 逐元素相乘 + 求和 与 拼接 + 线性变换是等价的

> ## **1️⃣ 原始 GAT 公式**
>
> 在 GAT 论文中，注意力分数 $e_{ij}$ 计算如下：
>
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
> ---
>
> ## **2️⃣ 逐元素相乘+求和的变形**
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
> ---
>
> ## **3️⃣ 等价性证明**
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
> ---
>
> ## **4️⃣ 结论**
>
> $$
> \mathbf{a}^T [W\mathbf{h}_i || W\mathbf{h}_j] = \sum_k a_{l,k} (W \mathbf{h}_i)_k + \sum_k a_{r,k} (W \mathbf{h}_j)_k
> $$
>
> 即 **逐元素相乘+求和的形式** 与 **拼接+矩阵乘法** 是等价的！
>
> ### **为什么这样实现更好？**
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
