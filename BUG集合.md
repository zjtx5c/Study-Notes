# BUG 集合以及调参、实验经验

## Hard Bug

* Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward

  尝试第二次向后遍历图（或者尝试访问在反向传播过程中保存的中间计算结果）。当您调用.backward（）或autograd.grad（）时，将释放图中保存的中间值。如果需要第二次向后遍历图，或者在调用backward后需要访问保存的张量，则指定retain_graph=True

  * 个人感觉有两个原因（1）连续两次调用 `.backward()`（2）尝试在反向传播过程中访问已经释放掉的中间结果。
  * 最简单暴力的解决办法就是 `loss.backward(retain_graph=True)`，只是对内存消耗大，对实际的结果并没有什么影响。
  * **经过研究，我发现了是哪个地方的梯度断了，使用代码进行详细排查**
  
    ```
    for name, param in model.named_parameters(): 
        print(name, param.requires_grad, param.grad)
    ```
    或输入以下代码进行快速排查
    ```
     for name, param in model.named_parameters():
        if param.grad is None:
            print(f"参数 {name} 的梯度是 None，计算图可能断裂")
    ```
    发现几乎都是 `gat_layers0` 或 `gat_layers1` 中的 `attn_l`, `attn_r`, `attn_m`, `.bias` 与 `fc.weight` 断裂 且都与**边嵌入**相关！！！！。于是顺腾摸瓜终于被我排查出了错误，原来是在 `MultiImportModel` 中传入 `edge_weight` 时候， `edge_weight` 并不是可学习的张量。解决方法为将原先`models` 模块中的 `edge_weight_dic = {etype: g.edges[etype].data['weight'] for etype in g.etypes}` 修改为 `edge_weight_dic = {etype: nn.Parameter(g.edges[etype].data['weight']) for etype in g.etypes}`
  * 花费时间 一天+一上午
  * ==后续：又在张量的问题上出错了，以后的实验一定要好好检查哪些应该被设置成**叶子张量**== （导致这点出现的原因是，我们忽略了我们的嵌入是自己构造的，而一般的嵌入都是通过 `word2vec` 等算法生成的，生成过程中会使用到 `nn.Module` 模块，它们会自动将对应的数据注册为叶子张量）。

## Easy Bug

* TypeError: torch.FloatTensor is not a Module subclass
  FloatTensor不是Module的子类。

  ```python
  layers = nn.ModuleList([  # 使用 ModuleList
      nn.Linear(10, 20),
      nn.ReLU(),
      nn.Linear(20, 1),
      torch.tensor(3.0)
  ])
  ```

  当我尝试进行如上操作时，被报错了。原因在于 `torch.tensor()` 并不是 `nn.Module` 子类，而是 `Tensor()` 的子类。`nn.ModuleList()` 中（只能包含）需要传入的是类似于 `nn.Linear()`,`nn.ReLU` 等 `nn` 。而这类东西它们的参数能够被自动注册且可以计算梯度，即能被`parameters()` 方法识别与优化器更新，但是 `Tensor` 类不具备这个功能。

  
  
  ```python
  layers = nn.ModuleList([  # 使用 ModuleList
      nn.Linear(10, 20),
      nn.ReLU(),
      nn.Linear(20, 1),
      nn.Parameter(torch.tensor(3.0))
  ])
  ```
  
  我在此进行了尝试，仍然收到了同上述一样的报错。**这是因为 `nn.Parameter` 并不是 `nn.Module`  的子类，`nn.Parameter` 只是一个可训练的 `Tensor`** 。`nn.Parameter` 是 **`torch.Tensor` 的子类**，它的唯一作用是**告诉 PyTorch 这个 `Tensor` 需要被训练**，**它不包含 `forward()` 逻辑，也不存储子模块，所以不是 `nn.Module`**。
  
  而 `nn.Module` 是一个容器：`nn.Module` 主要用于**管理** `nn.Linear()`、`nn.Conv2d()` 等**包含权重**的层，并且它可以**嵌套**其他 `nn.Module`，形成完整的模型。
  
  `nn.Module` 有 **`forward()` 方法**，可以定义前向传播的计算逻辑。
  
  `nn.Parameter` 只有作为 `nn.Module` 的 **属性**（比如 `self.param = nn.Parameter(...)`），才会被 `parameters()` 识别。
  
  如果 `nn.Parameter` **并未**添加到模型的 `parameters()` 中，而**只是作为中途计算的一部分**，它将 **不会** 被视为模型的可训练参数，且不会通过反向传播进行更新。它的效果实际上等同于一个普通的 **有梯度的 Tensor**，而不是一个真正的可训练参数。

* 发生异常: NotImplementedError exception: no description（太蠢了这个BUG）

  `NotImplementedError` 是 Python 中的一种错误，它通常表示某个方法或函数在代码中被声明了，但没有实现。也就是说，虽然该方法被调用了，但它的具体实现缺失了。

  ```python
  loss_1 = self.loss_eta1 * self.loss_1_fcn(embed_important, embed_normal)
  ```

  长话短说，报错的原因是我的 `self.loss_1_fcn` 的 `forward` 拼写错了（😂）

* 有一个在 `pandas` 的框架上经过筛选操作之后使用 `.reshape()` 之后，发现 `data.shape` 居然会有 0 维，而且 `data ` 居然是 `None` 。这是什么原因？

  条件筛选这一块出错了，经过筛选之后数据变空了...这个错误实在是太蠢了...





## 实验经验

### 早停与可视化

我们现在在训练一个模型，添加了 早停机制。但是我们又想在最后将 loss 和 score 可视化。我们原定的是训练200轮epoch，但是在75轮的时候早停了。该怎么做。是保存这个早停模型，接着跑到200轮还是直接用75轮的数据做loss和 score可视化。

推荐的做法是使用已有的75轮数据进行可视化。

原因一：早停就是为了防止过拟合

- 提前停止训练意味着模型在验证集上已经达到最佳表现，再继续训练反而可能会恶化性能；
- 再继续训练到200轮不仅浪费计算资源，还违背了早停机制的初衷。

原因二：可视化的目的之一就是**观察早停是否合理**

- 通过绘图可以看到验证损失在什么时候最小、是否出现过拟合；
- 使用 75 轮的数据画出 loss 曲线（train/val）和评价指标（score），反而更真实地反映模型训练过程。

