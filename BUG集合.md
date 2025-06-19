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
  
* 在复现 EASING 这个论文过程中。我发现 TMDB5K 仅在边数比 FB15K 大的情况下（所需显存大概是后者的四倍！）（未完，搁置）

  `FB15K` 数据情况：

  ```python
  ----Data statistics------'
            #Edges 592213
            #Unlabeled nodes 9883
            #Train samples 1407
            #Val samples 1411
            #Test samples 1409
  ```
  
  `TMDB5K` 数据情况：
  
  ```python
  ----Data statistics------'
            #Edges 761648
            #Unlabeled nodes 3365
            #Train samples 479
            #Val samples 480
            #Test samples 478
  ```
  
  可以发现在实验时使用的 `FB15K` 的节点数量大约是 `TMDB5K` 的 3倍。而 `TMDB5K` 的边数仅是 `FB15K` 边数的
  
  1.3 倍。但是，在训练参数完全一致的情况下，`FB15K` 没有爆显存，`TMDB5K` 居然爆显存了。这是为什么？
  
  省流版：归根结底一句话：虽然我们真正用于训练且被我们所关注的节点数量只有不到10000（这里分别是9883和3365），但是我们实际用于训练的节点与边却是**整张图的节点与边**，当然边也是整张图的边。`FB15K` 数据的节点和边分别是14,951 与 592,213。但是 TMDB5K 整张图的节点与边分别是114,805 和 761,648。后者的节点数量约是前者节点数量的 8 倍。IMDB就更夸张了，节点是10倍，边是2倍！问题就出在这里，显存直接膨胀。
  
  最根本的还是应该根据报错信息来。报错信息最终递归到这一行：
  
  ```python
  s, u = self.unc_trans_layers[l](s, u)
  ```
  
  说明是在解码的 forward 过程出问题了！对应代码中的这个部分：
  
  ```python
  def forward(self, s, u):
      qs = self.w_qs(s).reshape(s.shape[0],self.heads_num,s.shape[1],self.heads_dim) # [B, heads_num, N, heads_dim]
      ks = self.w_ks(s).reshape(s.shape[0],self.heads_num,s.shape[1],self.heads_dim) # [B, heads_num, N, heads_dim]
      vs = self.w_vs(s).reshape(s.shape[0],self.heads_num,s.shape[1],self.heads_dim) # [B, heads_num, N, heads_dim]
  
      qu = self.w_qu(u).reshape(s.shape[0],self.heads_num,s.shape[1],self.heads_dim) # [B, heads_num, N, heads_dim]
      ku = self.w_ku(u).reshape(s.shape[0],self.heads_num,s.shape[1],self.heads_dim) # [B, heads_num, N, heads_dim]
      vu = self.w_vu(u).reshape(s.shape[0],self.heads_num,s.shape[1],self.heads_dim) # [B, heads_num, N, heads_dim]
  
      ua_s_attn = torch.matmul(qs / self.sqrt_d, ks.transpose(-2,-1)) # [B, heads_num, N, N]
      ua_s_attn = F.softmax(ua_s_attn,dim=-1) # [B, heads_num, N, N]
      ua_s = torch.matmul(ua_s_attn, vs).view(s.shape[0],s.shape[1],s.shape[2]) # [B, N, 2d]
  
      ua_u_attn = torch.matmul(qu / self.sqrt_d, ku.transpose(-2,-1))
      ua_u_attn = F.softmax(ua_u_attn,dim=-1) # [B, heads_num, N, N]
      ua_u = torch.matmul(ua_u_attn, vu).view(u.shape[0],u.shape[1],u.shape[2]) # [B, N, 2d]
  
      s_hat = self.layer_norm_s_1(s + ua_s) # [B, N, 2d]
      s_output = self.layer_norm_s_2(s_hat + self.FFN_s(s_hat)) # [B, N, 2d]
  
      u_hat = self.layer_norm_u_1(u + ua_u) # [B, N, 2d]
      u_output = self.layer_norm_u_2(u_hat + self.FFN_u(u_hat)) # [B, N, 2d]
  
      return s_output, u_output
  ```
  
  这里的 `B` 就是整张图的节点数量。其实在原本的解码阶段显存是能够过的（将 `hidden_dim` 设置为 8 的情况下），但是在解码阶段凭空又多出了一个维度，并且按照附录的设置 `10` 就爆炸了。现在来计算一下到底差了多少。
  
  从节点角度看就是直接差8倍。粗模地计算了一层 `DJE layer`的显存大概是 600mb+（TMDB5K 数据）
  
  > | 模块                | 估算显存占用 |
  > | ------------------- | ------------ |
  > | Linear 输入输出 ×6  | ~220 MB      |
  > | Attention matrix ×2 | ~175 MB      |
  > | FFN + 残差输出 ×2   | ~220 MB      |
  > | 总计                | **~615 MB**  |
  
  实际上我们有两层 DJE layer 且有两个模型加入训练，也就是说我们解码的显存大概有 615MB * 4 = 2.4GB.当然我们没有将 `backward、optimizer` 计算进去（这种杂七杂八的加进去大概也是 2.4GB）所以解码部分至少也需要4.8GB.（这个计算逻辑是合理的，因为我拿分别拿 d = 8 和 d = 16 代入验算了一下是符合实际情况的）。
  
  若将 $d$ 改成 4 呢？感觉显存是够用的呀？粗略地算了一下，解码部分大概只需要 2.3GB 显存了。应该是可以了呀。。。还是不行。
  
  稍微计算了一下，编码部分的显存大概为解码部分显存的 $\frac{1}{N}$ 而这里 $N = 10$
  
  考虑的改进办法
  
  1. 使用小 `batch` 分批处理
  2. 使用 `float16` 精度
  3. 缩小 `N` 的规模
  
  

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

