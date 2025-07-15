# Transfromers_T5

Bert 是只有编码部分，GPT 是只有解码部分，而 T5 涵盖编码与解码部分。

T5（**Text-To-Text Transfer Transformer**）的中文全称通常翻译为：

> 文本到文本的迁移式Transformer模型

它是一个大小写敏感的模型



## 参考文档

*  https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html

- https://huggingface.co/docs/transformers/model_doc/t5



## 模型概览

### 任务

![T5](C:\Users\5c\Desktop\Study Notes\pic\T5.png)

可以发现，T5是一个强大的模型，我们对其指定一些评估任务，它会进行编码与解码。

* translate：翻译

* cola sentence：

  > 英文全称：**Corpus of Linguistic Acceptability**
  >
  > **中文解释：语言可接受性语料库**
  >
  > **任务目标**：判断一句话是否是“语法上正确”的英语句子。
  >
  > - 输入：一个句子
  > - 输出：0（不可接受）或 1（可接受）

* stsb sentence1：

  > 英文全称：**Semantic Textual Similarity Benchmark**
  >
  > 中文解释：**语义文本相似度基准任务**
  >
  > **任务目标**：给定两个句子，判断它们的语义相似度（从 0 到 5 的分值）
  >
  > - 输入：一对句子
  > - 输出：一个连续值（通常是浮点数），表示它们的语义相似程度

* summarize：总结



### 我们以 T5-small 模型为例

其参数量大约位61M，里面的 `hidden_dim` 大概是 `512 (64\*8)  ->  512`，共有 12 层，T5 其实是 `block` 而非 `layer`。

#### encoder

我们以该模型的第一层（`block[0]`） 为例来理解一下它的架构：

```bash
(0): T5Block(
      (layer): ModuleList(
        (0): T5LayerSelfAttention(
          (SelfAttention): T5Attention(
            (q): Linear(in_features=512, out_features=512, bias=False)
            (k): Linear(in_features=512, out_features=512, bias=False)
            (v): Linear(in_features=512, out_features=512, bias=False)
            (o): Linear(in_features=512, out_features=512, bias=False)
            (relative_attention_bias): Embedding(32, 8)
          )
          (layer_norm): T5LayerNorm()
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (1): T5LayerFF(
          (DenseReluDense): T5DenseActDense(
            (wi): Linear(in_features=512, out_features=2048, bias=False)
            (wo): Linear(in_features=2048, out_features=512, bias=False)
            (dropout): Dropout(p=0.1, inplace=False)
            (act): ReLU()
          )
          (layer_norm): T5LayerNorm()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
```

很显然，它和一般的 transformer 架构一样都是由一个 **attention 层 + 一个前馈神经网络层（FF，feedforward）构成。**当然，也还包括一个**后处理部分**。

当然，还值得注意的是 attention 部分，还有一个 `o` 模块，它再次将一个 512 维度的特征映射到一个 512 维度的特征，其目的是为了融合所有头学习到的“知识”。

FF部分就是

```bash
wi: Linear(512 → 2048)    # 第一个全连接层，升维
act: ReLU()               # 激活函数
wo: Linear(2048 → 512)    # 第二个全连接层，降维
```

最后就是后处理部分了，还是比较好理解的



#### decoder

```bash
(block): ModuleList(
    (0): T5Block(
      (layer): ModuleList(
        (0): T5LayerSelfAttention(
          (SelfAttention): T5Attention(
            (q): Linear(in_features=512, out_features=512, bias=False)
            (k): Linear(in_features=512, out_features=512, bias=False)
            (v): Linear(in_features=512, out_features=512, bias=False)
            (o): Linear(in_features=512, out_features=512, bias=False)
            (relative_attention_bias): Embedding(32, 8)
          )
          (layer_norm): T5LayerNorm()
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (1): T5LayerCrossAttention(
          (EncDecAttention): T5Attention(
            (q): Linear(in_features=512, out_features=512, bias=False)
            (k): Linear(in_features=512, out_features=512, bias=False)
            (v): Linear(in_features=512, out_features=512, bias=False)
            (o): Linear(in_features=512, out_features=512, bias=False)
          )
          (layer_norm): T5LayerNorm()
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (2): T5LayerFF(
          (DenseReluDense): T5DenseActDense(
            (wi): Linear(in_features=512, out_features=2048, bias=False)
            (wo): Linear(in_features=2048, out_features=512, bias=False)
            (dropout): Dropout(p=0.1, inplace=False)
            (act): ReLU()
          )
          (layer_norm): T5LayerNorm()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
```

唯二有变数的是加了解码器较于编码器多了一个 `T5LayerCrossAttention` 交叉注意力层，其关键点是：

| 输入                 | 来自哪里                                             |
| -------------------- | ---------------------------------------------------- |
| Query `Q`            | 来自 decoder 自己的 hidden states（当前 token 表示） |
| Key `K` 和 Value `V` | 来自 encoder 的输出（input_ids 编码结果）            |

当然还有一个 `(relative_attention_bias): Embedding(32, 8)` 这是相对位置编码（搁置）



### 模型规模

- vocabulary size：32128

|model        |参数量       |hidden dim                |  encoder/decoder layers | 
| ----------- |----------- |------------------------- | ------------------------| 
|t5-small     | 61M        |     512 (64\*8)  ->  512 |                        6|
|t5-base      |223M        |    768  (64\*12) ->  768 |                       12|
|t5-large     |738M        |   1024  (64\*16) -> 1024 |                       24|
|t5-3b        |2.85B       |   4096 (128\*32) -> 1024 |                       24|
|t5-11b       |  11B       | 16384 (128\*128) -> 1024 |                       24|



## forward

```python
input_ids = tokenizer(
    "Studies have been shown that owning a dog is good for you", return_tensors="pt"
).input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  
# preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
# This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
decoder_input_ids = model._shift_right(decoder_input_ids)
```

* T5 模型有两个版本：

  > - `T5Model`: 基础版本，需要你**手动处理 decoder 的输入**
  > - `T5ForConditionalGeneration`: 处理了 decoder 输入的预处理（如 `shift_right`）
  >
  > 对于 `T5Model`，我们必须自己“右移（shift right）” decoder 的输入序列，加上起始 token

* ```python
  decoder_input_ids = model._shift_right(decoder_input_ids)
  ```

  上面的这份代码有什么用？

  > 调用 T5 模型的私有方法 `_shift_right`，将 decoder 的输入向右移动 1 格，并在开头加上起始 token（通常是 `<pad>` token 的 id，即在 T5 模型中，起始符号是 `<pad>`）。
  >
  > 这是训练 decoder 的标准操作，目的是：
  >
  > > 让 decoder 在第 `i` 个时间步预测第 `i+1` 个词（teacher forcing）。
  >
  > 举个例子：
  >
  > 假设你原来的 decoder 输入是：
  >
  > ```bash
  > "Studies show that"
  > => [token1, token2, token3]
  > 
  > ```
  >
  > `_shift_right()` 之后会变成：
  >
  > ```bash
  > [<pad>, token1, token2]
  > ```
  >
  > 也就是让 decoder 每次看到上一个 token 来预测当前 token（自回归地生成）。



* 我们可以尝试自己实现一下它的“前向”过程，当然，核心代码是扒的。

  ```python
  def t5_forward(model, input_ids, decoder_input_ids):
      encoder_outputs = model.encoder(input_ids=input_ids)
      print(encoder_outputs)
      hidden_states = encoder_outputs[0]
      decoder_outputs = model.decoder(input_ids=decoder_input_ids, 
                                      encoder_hidden_states=hidden_states,)
      return decoder_outputs.last_hidden_state
  ```

  这里的参数顺序与位置其实还不是很懂，等学了 GPT 应该就会懂一点了。

  

  

## 预训练任务

- Unsupervised denoising training
    - MLM
    
      > 随机 mask 掉一些 token，让模型预测这些位置的词是什么
      >
      > ```bash
      > 原句： The cat sat on the mat.
      > MLM输入： The [MASK] sat on the [MASK].
      > 目标：    cat          mat
      > ```
      >
      > **模型学习的是词级上下文推理能力**
    
    - span mask
    
      > 不是只 mask 单个词，而是 **mask 一段连续 token（一个 span）**
      >
      > ```bash
      > 原句： The cat sat on the mat.
      > Span mask后： The <extra_id_0> on the <extra_id_1>.
      > 目标： <extra_id_0>: cat sat
      >        <extra_id_1>: mat
      > ```
      >
      > T5 中 `<extra_id_0>` 是特殊的标记，用来替代被 mask 的 span。
      >
      > 优点：
      >
      > - 更接近真实的生成任务
      > - 更适合 seq2seq 框架
    
- Supervised training
    - seq2seq
    
      > 输入是一个序列，输出也是一个序列
      >
      > 是 Encoder-Decoder 结构
      >
      > **例子：机器翻译任务**
      >
      > ```bash
      > 输入：Translate English to French: I love dogs.
      > 输出：J'aime les chiens.
      > ```
      >
      > **例子：摘要任务**
      >
      > ```bash
      > 输入：Summarize: The US president gave a speech on climate...
      > 输出：President speaks on climate.
      > ```

| 阶段   | 类型   | 技术      | 模型代表 | 是否需要标注 |
| ------ | ------ | --------- | -------- | ------------ |
| 预训练 | 无监督 | MLM       | BERT     | ❌            |
| 预训练 | 无监督 | Span Mask | T5, BART | ❌            |
| 微调   | 有监督 | Seq2Seq   | T5, BART | ✅            |



### 无监督学习——mlm

```bash
# Unsupervised denoising training
# mlm
input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids

# the forward function automatically creates the correct decoder_input_ids
loss = model(input_ids=input_ids, labels=labels).loss
loss.item()
```

我们可以发现 `input_ids` 和 `labels` 是互补的！

| input_ids                       | labels                       |
| ------------------------------- | ---------------------------- |
| 留下未删内容 + span占位符       | 包含被删的内容 + 占位提示    |
| "The <0> walks in the <1> park" | "<0> cute dog <1> sunny <2>" |
| 把原句拆成两部分                | 把两部分拼回去               |

换句话说：input_ids + labels 的内容总和 = 原始完整句子。所这是一个预训练过程。

具体的训练流程是怎样的呢？



### 监督学习——seq2seq

```python
# Supervised training
# seq2seq

input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids

# the forward function automatically creates the correct decoder_input_ids
loss = model(input_ids=input_ids, labels=labels).loss
loss.item()
```

为什么模型会知道它要做的任务，我们明明没有给出对应的任务参数。它是如何“知道要翻译”？

> 是你**手动告诉它的**，这部分就在这里：
>
> ```bash
> "translate English to German:"
> ```
>
> 这段是 T5 设计中的核心思想：
>
> > **T5 把“任务类型”也当作输入的一部分，嵌入进句子中当作 Prompt。**
>
> 这就是 T5 的理念：**Text-To-Text Transfer Transformer**
>
> - 所有任务都被转换为一个文本输入 → 文本输出的问题
> - 不同任务，通过“指令文本”区分

| 任务类型 | 输入（input_ids）                    |
| -------- | ------------------------------------ |
| 翻译     | `translate English to German: ...`   |
| 文本摘要 | `summarize: ...`                     |
| 文本分类 | `sst2 sentence: this movie is great` |
| 问答     | `question: ... context: ...`         |

所以在 T5 的语义里：

- “你是什么任务” ≈ “你怎么写 prompt”
- T5 不自动判断任务，你得告诉它



### **multi sentence pairs**

```python
# the following 2 hyperparameters are task-specific
max_source_length = 512
max_target_length = 128

# Suppose we have the following 2 training examples:
input_sequence_1 = "Welcome to NYC"
output_sequence_1 = "Bienvenue à NYC"

input_sequence_2 = "HuggingFace is a company"
output_sequence_2 = "HuggingFace est une entreprise"

# encode the inputs
task_prefix = "translate English to French: "
input_sequences = [input_sequence_1, input_sequence_2]

encoding = tokenizer(
    [task_prefix + sequence for sequence in input_sequences],
    padding="longest",
    max_length=max_source_length,
    truncation=True,
    return_tensors="pt",
)

input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

# encode the targets
target_encoding = tokenizer(
    [output_sequence_1, output_sequence_2],
    padding="longest",
    max_length=max_target_length,
    truncation=True,
    return_tensors="pt",
)
labels = target_encoding.input_ids

# replace padding token id's of the labels by -100 so it's ignored by the loss
labels[labels == tokenizer.pad_token_id] = -100

# forward pass
loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
loss.item()
```

整个流程干了什么？

| 步骤            | 内容                                           |
| --------------- | ---------------------------------------------- |
| 准备数据        | 2 对 英文 → 法文 句子                          |
| 拼接 **prompt** | 每条输入前加 `"translate English to French: "` |
| tokenizer 编码  | 把**输入**和**输出**都变成 token id 张量       |
| 设置 label mask | **忽略 pad 的损失计算**                        |
| 模型 forward    | 用 T5 执行 seq2seq 训练，计算 loss             |



## 完成 task

```python
input_ids = tokenizer.encode("translate English to German: Hello, my dog is cute", return_tensors="pt") 
# 使用推理模式中的生成方法
result = model.generate(input_ids)
tokenizer.decode(result[0])
```





```python
# inference
input_ids = tokenizer(
    "summarize: Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts every language problem into a text-to-text format. Our systematic study compares pretraining objectives, architectures, unlabeled datasets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new “Colossal Clean Crawled Corpus”, we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our dataset, pre-trained models, and code.", 
    return_tensors="pt"
).input_ids  # Batch size 1
outputs = model.generate(input_ids, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# studies have shown that owning a dog is good for you.
```

| 方法                        | 用途                                                         | 输出                            |
| --------------------------- | ------------------------------------------------------------ | ------------------------------- |
| `model(input_ids)`          | 用于训练，计算损失。返回每个 token 的 logits（预测概率）     | logits（没有生成文本）          |
| `model.generate(input_ids)` | 用于**推理**，生成文本。自动回归地生成下一个 token，直到完成文本生成。 | 生成的 token 序列（生成的文本） |

