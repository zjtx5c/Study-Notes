# Transformers

## 01_fine_tune_transformers_on_classification

这一节硬货太多了，得沉下心来慢慢消化。

### 要点与疑问

* 可视化过程中，我们重点分析文本长度与类别频率
* 学会在 huggingface 上下载和上传

* `AutoModel` 和 `AutoModelForSequenceClassification` 的区别

  | 特性         | `AutoModel`                                      | `AutoModelForSequenceClassification`                         |
  | ------------ | ------------------------------------------------ | ------------------------------------------------------------ |
  | **目标任务** | 通用的编码器模型，适用于多种任务（无任务特定层） | 专用于序列分类任务，包含分类头（如线性层）                   |
  | **模型结构** | 只有编码器部分，没有任务特定的头（如分类头）     | 包含了分类头（通常是一个线性层），用于分类任务               |
  | **用途**     | 用于获取编码器输出或进行自定义任务               | 用于文本分类任务（如情感分析、新闻分类等）                   |
  | **输出**     | 输出的是模型的隐藏状态或嵌入向量                 | 输出的是分类结果（logits，经过分类头的输出）                 |
  | **加载方式** | `AutoModel.from_pretrained('model_name')`        | `AutoModelForSequenceClassification.from_pretrained('model_name')` |

### **text classification**（文本分析）

- 也叫 sequence classification
- sentiment analysis
    - 情感分析，就是一种文本/序列分类
        - 电商评论
        - social web：weibo/tweet

#### emotions 数据集

理解获取 hugging face 数据集的常用组织结构

**Hugging Face `datasets`库**中**绝大多数数据集**（包括你自己用 Python 构建的 `Dataset`）都是采用**这种结构组织的**：

> 划分话训练集、验证集与测试集（8：1：1）
>
> **一组具有相同字段（features）和相同行数的数据列**，每一行是一个样本，每一列是一个字段（例如 text、label）。
>
> ```bash
> DatasetDict({
>     train: Dataset({
>         features: ['text', 'label'],
>         num_rows: 16000
>     })
>     validation: Dataset({
>         features: ['text', 'label'],
>         num_rows: 2000
>     })
>     test: Dataset({
>         features: ['text', 'label'],
>         num_rows: 2000
>     })
> })
> ```
>



#### tokenize the whole dataset

学习一下这里的优雅写法：内容包括 `map()` 函数（事实上这个是 `datasets` 库中的 `Dataset.map()` 方法） + 对迭代器的理解

* `map(function, iterable)`

  > `function`：你想对每个元素应用的函数
  >
  > `iterable`：可迭代对象（如列表、元组等）

我们考虑对 `emotions` 数据集中的 `train` `val` `test` 数据集都进行 `tokenize`，考虑一下该怎么写？首先需要明确的是，`emotions` 是一个可迭代对象。那么我们就可以使用 `map` 这个方法了

事实上，**是在使用 🤗 Hugging Face 的 `datasets` 库 中的 `Dataset.map()` 方法，对一个文本数据集 `emotions` 进行批处理式的预处理或 tokenization。**

```python
emotions
for e in emotions:
    print(e)
```

```bash
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 16000
    })
    validation: Dataset({
        features: ['text', 'label'],
        num_rows: 2000
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 2000
    })
})

train
validation
test
```



我们使用 `datasets` 库中的 `Dataset.map()` 方法，进行批处理式的预处理或 `tokenization`

```python
def batch_tokenize(batch):
    return tokenizer(batch["text"], padding = True, truncation = True)
    
emotions_encoded = emotions.map(batch_tokenize, batched=True, batch_size = None)
emotions_encoded
```

```bash
DatasetDict({
    train: Dataset({
        features: ['text', 'label', 'input_ids', 'attention_mask'],
        num_rows: 16000
    })
    validation: Dataset({
        features: ['text', 'label', 'input_ids', 'attention_mask'],
        num_rows: 2000
    })
    test: Dataset({
        features: ['text', 'label', 'input_ids', 'attention_mask'],
        num_rows: 2000
    })
})
```

> `batch_tokenize`
>
> - 是你自己定义的函数，或用 tokenizer 包装过的函数；
> - 用来对一批文本（如 `batch["text"]`）进行处理，例如 tokenizer：
>
> ```python
> def batch_tokenize(batch):
>     return tokenizer(batch["text"], padding=True, truncation=True)
> ```
>
> `batched=True`
>
> - 表示 `batch_tokenize` 是**批处理函数**；
> - Hugging Face 会把数据分成小批次（默认 1000 条），然后传给你的函数；
> - 每次传入的 `batch` 是一个字典，例如：
>
> ```python
> {
>   "text": [...1000 条文本...],
>   "label": [...对应标签...]
> }
> ```
>
> `batch_size=None`
>
> - 表示使用默认批大小（大约 1000 条）；
> - 你也可以写 `batch_size=32` 之类的。



另外还有
```python
emotions_encoded.set_format('torch', columns=['label', 'input_ids', 'attention_mask'])
```

是 Hugging Face `datasets` 库中 `Dataset` 对象的方法，用于将指定的列转换为 **PyTorch 张量（tensor）格式**，方便用于模型训练。（**原来一般是 list 类型**）

> 把 `emotions_encoded` 数据集中的 `'label'`, `'input_ids'`, `'attention_mask'` 三列转换成 `torch.Tensor` 类型，以便和 PyTorch 模型对接。
>
> 我们不需要对 `text` 进行 tensor 类型的转换，因为它不是我们训练的对象，也可以认为它是给我们看的而不是给机器看的。





### **fine-tune transformers**

#### distilbert-base-uncased

* `distilbert` 是对 `bert` 的 `distill` 而来
  * 模型结构更为简单，
  * `bert-base-uncased` 参数量：109482240
  * `distilbert-base-uncased` 参数量：66362880

可以比较一下它相较于 Bert，少了哪些东西。

我们导入一个做下游任务的模型

```python
from transformers import AutoModelForSequenceClassification
model_ckpt = 'distilbert-base-uncased'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_classes).to(device)
model
```

观察其 `model` 结构

```bash
DistilBertForSequenceClassification(
  (distilbert): DistilBertModel(
    (embeddings): Embeddings(
      (word_embeddings): Embedding(30522, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (transformer): Transformer(
      (layer): ModuleList(
        (0-5): 6 x TransformerBlock(
          (attention): MultiHeadSelfAttention(
            (dropout): Dropout(p=0.1, inplace=False)
            (q_lin): Linear(in_features=768, out_features=768, bias=True)
            (k_lin): Linear(in_features=768, out_features=768, bias=True)
            (v_lin): Linear(in_features=768, out_features=768, bias=True)
            (out_lin): Linear(in_features=768, out_features=768, bias=True)
          )
          (sa_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (ffn): FFN(
            (dropout): Dropout(p=0.1, inplace=False)
            (lin1): Linear(in_features=768, out_features=3072, bias=True)
            (lin2): Linear(in_features=3072, out_features=768, bias=True)
            (activation): GELUActivation()
          )
          (output_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        )
      )
    )
  )
  (pre_classifier): Linear(in_features=768, out_features=768, bias=True)
  (classifier): Linear(in_features=768, out_features=6, bias=True)
  (dropout): Dropout(p=0.2, inplace=False)
)
```

可以发现，前面就是一个 `distilbert-base-uncased`，而后面则是一个下游任务的 `head`，需要注意的是。它会返回一个提示

> ```bash
> Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['pre_classifier.bias', 'pre_classifier.weight', 'classifier.weight', 'classifier.bias']
> You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
> ```

`['pre_classifier.bias', 'pre_classifier.weight', 'classifier.weight', 'classifier.bias']` 这**三个下游层的模型是没有训练过的！！** 

我们需要 `fine tune`



#### triner

> `transformers.Trainer` 是 Hugging Face 提供的一个**通用训练循环封装器**，内部集成了：
>
> - 数据加载（DataLoader）
> - 前向传播 & 损失计算
> - 反向传播 & 梯度裁剪
> - Optimizer & Scheduler 更新
> - 模型保存、评估、日志记录
> - 支持多 GPU / 混合精度训练（FP16）等功能



使用示例

```python
from transformers import TrainingArguments, Trainer
batch_size = 16  # 每个设备上的训练/验证 batch 大小设为 16
logging_steps = len(emotions_encoded['train']) // batch_size  # 每个 epoch 日志记录一次

model_name = cache_dir_model + f'{model_ckpt}_emotion_ft_0416'  # 模型输出路径，命名包含模型名和日期标识

training_args = TrainingArguments(
    output_dir=model_name,                          # 模型保存目录
    num_train_epochs=4,                             # 总训练轮数为 4
    learning_rate=2e-5,                             # 初始学习率设为 2e-5（常用于 BERT 微调）
    weight_decay=0.01,                              # 权重衰减系数，防止过拟合
    per_device_train_batch_size=batch_size,         # 每个设备的训练 batch 大小
    per_device_eval_batch_size=batch_size,          # 每个设备的验证 batch 大小
    evaluation_strategy="epoch",                    # 每个 epoch 结束后进行一次评估
    disable_tqdm=False,                             # 显示 tqdm 进度条
    logging_steps=logging_steps,                    # 每个 epoch 日志输出一次
    push_to_hub=True,                               # 训练完自动将模型推送到 Hugging Face Hub（需登录）
    log_level="error"                               # 日志等级设置为 error，仅输出错误信息
)

```



- trainer默认自动开启 torch 的多gpu模式，
    - `per_device_train_batch_size`: 这里是设置每个gpu上的样本数量，
    - 一般来说，多gpu模式希望多个gpu的性能尽量接近，否则最终多gpu的速度由最慢的gpu决定，
        - 比如快gpu 跑一个batch需要5秒，跑10个batch 50秒，慢的gpu跑一个batch 500秒，则快gpu还要等慢gpu跑完一个batch然后一起更新weights，速度反而更慢了。
    - 同理 `per_device_eval_batch_size` 类似
- `learning_rate`/`weight_decay`
    - 默认使用 AdamW 的优化算法



我们设置好了 `training_args ` 参数管理器之后，直接设置训练器。需要记住的是我们需要传入以下东西

1. 模型`model`
2. 分词器 `tokenizer`
3. 数据 `train_dataset` 与 `eval_dataset`
4. 超参数 `args = training_args`
5. 指定用于在验证集上计算评估指标的函数 `compute_metrics`

例如：

```python
from transformers_utils import compute_classification_metrics	# 这个是自定义的一个 工具包
trainer = Trainer(model=model, 
                  tokenizer=tokenizer,
                  train_dataset=emotions_encoded['train'],
                  eval_dataset=emotions_encoded['validation'],
                  args=training_args, 
                  compute_metrics=compute_classification_metrics)
```

我们重点理解一下这个 `compute_metrics`，一般格式如下：

```python
def compute_classification_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        ...
    }

```

`trainer` 会在每次 `evaluate()` 或每个 `epoch` 后，**调用这个函数来打印/记录指标**

Hugging Face 中 `compute_metrics(pred)` 的参数 `pred` 是一个 `EvalPrediction` 对象，其本质结构如下：

```python
EvalPrediction = namedtuple("EvalPrediction", ["predictions", "label_ids"])
```

而 `trainer.predict()` 返回的 `PredictionOutput` 结构如下：

```python
PredictionOutput = namedtuple("PredictionOutput", ["predictions", "label_ids", "metrics"])
```

你会发现它们结构是高度重合的，差别只在于：

| 字段        | EvalPrediction（评估用）  | PredictionOutput（预测用） |
| ----------- | ------------------------- | -------------------------- |
| predictions | ✅                         | ✅                          |
| label_ids   | ✅                         | ✅                          |
| metrics     | ❌（compute_metrics 输出） | **✅（调用后返回）**        |

==**这说明：`PredictionOutput` 其实是 `EvalPrediction` + `compute_metrics` 的输出结果。**==

可算是理解了



##### Hugging Face 的 Trainer，在每次训练（`.train()`）、评估（`.evaluate()`） 或预测（`.predict()`） 之后会返回什么数据类型

> ✅ `trainer.train()` 的返回值
>
> ```python
> train_output = trainer.train()
> ```
>
> 返回值是一个 **`transformers.TrainOutput` 对象**，你可以当成一个 dict-like 结构看待，它包含：
>
> ```python
> TrainOutput(
>     global_step=1234,
>     training_loss=0.4567,
>     metrics={
>         'train_runtime': 342.45,
>         'train_samples_per_second': 34.5,
>         'train_steps_per_second': 3.6,
>         'total_flos': 123456789.0,
>         'train_loss': 0.4567,
>         ...
>     }
> )
> ```
>
> 你可以通过：
>
> ```python
> train_output.metrics["train_loss"]  # 或
> train_output.global_step
> ```
>
> 访问这些内容。
>
> 
>
>  ✅ `trainer.evaluate()` 的返回值
>
> ```python
> eval_metrics = trainer.evaluate()
> ```
>
> 返回的是一个 **标准 Python 字典 `dict`**，内容由你提供的 `compute_metrics` 决定，可能包括：
>
> ```python
> {
>     'eval_loss': 0.2235,
>     'eval_accuracy': 0.897,
>     'eval_f1': 0.89,
>     'eval_runtime': 45.1,
>     'eval_samples_per_second': 100.2,
>     'epoch': 1.0
> }
> ```
>
> 说明：
>
> - `eval_loss` 是交叉熵或你定义的 loss
> - 其他 `eval_*` 项来自你的 `compute_metrics` 函数返回
> - `eval_runtime` 等为 Trainer 自动记录的效率指标
>
> 
>
> ✅ `trainer.predict()` 的返回值
>
> ```python
> predictions = trainer.predict(test_dataset)
> ```
>
> 返回的是一个 **`PredictionOutput` 命名元组**，结构如下：（注意和上文结合起来理解）
>
> ```python
> PredictionOutput(
>     predictions=array([...]),     # 通常是 logits
>     label_ids=array([...]),       # 真实标签
>     metrics={'test_accuracy': 0.88, 'test_f1': 0.86, ...}
> )
> ```
>
> 你可以访问：
>
> ```python
> predictions.predictions   # 模型输出的 logits
> predictions.label_ids     # 原始标签
> predictions.metrics       # compute_metrics 返回的字典
> ```
>
> 这里的结构和传入的 `compute_metric` 计算逻辑差不多
>
> ✅ 总结对比表：
>
> | 函数                 | 返回类型                         | 主要内容                            |
> | -------------------- | -------------------------------- | ----------------------------------- |
> | `trainer.train()`    | `TrainOutput`                    | global_step, training_loss, metrics |
> | `trainer.evaluate()` | `dict`                           | loss + 评估指标                     |
> | `trainer.predict()`  | `PredictionOutput`（namedtuple） | predictions + labels + metrics      |



##### 对验证集进行批量前向传播

学会在 hugging-face 中使用批量训练，我们举一个例子

```python
from torch.nn.functional import cross_entropy
def forward_pass_with_label(batch):
    # 将所有的和训练有关的输入转换成 tensor，比如 input_ids
    inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_name}

    with torch.no_grad():
        output = model(**input) # 会返回 logits 和 loss 之类
        pred_label = torch.argmax(output.logits, dim = -1)
        loss = cross_entropy(output.logits, batch["label"].to(device),
                             reduction = "none")
    
    return {"loss": loss.cpu().numpy(),
            "predicated_label": pred_label.cpu().numpy()}
    
    
    
emotions_encoded["validation"] = emotions_encoded["validation"].map(
    forward_pass_with_label, batched=True, batch_size=16)
emotions_encoded['validation']

```

```bash
Dataset({
    features: ['text', 'label', 'input_ids', 'attention_mask', 'loss', 'predicted_label'],
    num_rows: 2000
})
```



#### to huggingface hub

传到 hugging-face 上



## 02_transformer_architecture_self_attention

* [链接](https://www.bilibili.com/video/BV14s4y127kk?spm_id_from=333.788.videopod.sections&vd_source=56ba8a8ec52809c81ce429c827dc30ab)

区分 self-attention 中什么是 attention-weights ，什么是 attention-scores

q, k, v 的来源可以认为是信息检索系统

其他的直接看 book 就行了



* 计算 `scaled_dot_product_attention` 的做法（缩放点积注意力，是 Transformer 模型中的核心机制之一）

  ```python
  # batch_size, seq_len, hidden_size
  def scaled_dot_product_attention(query, key, value):
      # hidden_size
      dim_k = key.size(-1)
      # batch_size, seq_len, seq_len
      attn_scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(dim_k)
      attn_weights = F.softmax(attn_scores, dim=-1)
      return torch.bmm(attn_weights, value)
  ```



### summary

- attention mechanism
- encoder vs. decoder
    - seq2seq
        - seq of tokens (input) => seq of tokens (output)
        - 注意，这两个 tokens 是不一样的，一个针对 input 一个针对 output
    - tasks
        - machine translation
    - encoder
        - seq of tokens => seq of embedding vectors(hidden state/context)
    - decoder （**这个后续再看，现在还不是很懂**）
        - encoder's hidden state => seq of tokens
            - iteratively generate（迭代式的 generate）
                - until EOS (end of seq) or reach max length limit
                - one token at a time（每次只生成一个 token）



### encoder - decoder

- encoder only: seq of text => rich representation (bidirectional attention)
    - task（边界越来越模糊了，目前基本 GPT Bert T5 这些都能做）
        - text classification
        - NER
    - models
        - BERT
        - RoBERTa
        - DistilBERT
    - 完形填空（bidirectional）基于上下文去输出当前的一个词
        - representation of a given token depends both on
            - left (before the token)
            - right (after the token)
- decoder only (causal or autoregressive attention)自回归，当前的输出依赖于过去的输出
    - gpt
    - 词语接龙
        - representation of a given token depends only on the left context；（只依赖于之前的）
- encoder-decoder both
    - tasks
        - machine translation
        - summarization
    - models
        - t5
        - bart



### encoder

- encoder layer: encoding the contextual information (conv) (可以理解为 cv 中的卷积)
    - input: seq of embeddings
        - multi-head self attention
        - ffn(fc)
    - output: 
        - same shape as `input`
    - contextual information (contextualized embeddings )（语境化嵌入）
        - apple: company-like or fruit-like ?
           - keynote/phone/Jobs
           - banana, food, fruit
        - flies
           - time flies like an arrow：soars
           - fruit flies like a banana：insect
- skip connection (residual connection) & layer normalization
    - 高效训练深度神经网络的技巧；



### self attention

- each token (embedding) (在经过 self attention 之后)
    - 不是 fixed embeddings 
    - 而是 weighted average of each embedding of the whole input sequence（**和上下文语境有关的嵌入了（加权），不再是一个固定的 embedding 了**）
- a seq of token embedding：$x_1, x_2, \cdots, x_n$，经过 self attention 得到 a seq of updated embeddings，$x'_1, x'_2, \cdots, x'_n$

$$
x'_i=\sum_{j=1}^n w_{ji}x_j
$$

- $w_{ji}$
    - attention weights, 
    - $\sum_{j}w_{ji}=1$
    - $w\in\mathcal R^{n\times n}$，方阵，$seq_{len} \times seq_{len}$



**训练好的嵌入是具有上下文语境的嵌入，具体来说是上下文其他词嵌入的加权和，这个权就是注意力权重**

- Project each token embedding into three vectors called query, key, and value.
    - W_q, W_k, W_v：learnable parameters
- Compute attention **scores**. 
    - dot-product(**query, key**) => attention scores;
    - a sequence with $n$ input tokens there is a corresponding $\mathcal R^{n\times n}$ matrix of attention scores.
- Compute attention weights（$w_{j,i}$） from attention scores
    - Dot products 的结果可能是任意大的数，会让整个训练过程非常不稳定
    - 将 attention scores 乘以一个 scaling factor 
    - softmax 归一化：$\sum_{j}w_{ji}=1$
- update the final embedding of the token (**value**)
    - $x'_i=\sum_{j=1}^n w_{ji}v_j$
