# BUG 集合

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

* ```python
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
  
  

