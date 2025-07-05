# transfroms

## 安装问题

我们需要使用 hugging face，但是被墙，即便科学上网也无法安装模型。具体解决办法可以查看[镜像网站](https://hf-mirror.com/)的教程。我已经设置为全局镜像了，但好像没有。

~~感觉只有手动下载了~~

终于行了！

解决方法是在导入相关的 `transformers` **之前**加上这两句代码即可解决问题

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

以下是原因解释：

> 这是因为 Hugging Face 的 `transformers` 库在**首次导入时**，就会读取环境变量来确定你使用的镜像或访问地址。
>
> 🔍 解释原因：
>
> 当你执行：
>
> ```python
> from transformers import AutoTokenizer, AutoModel
> ```
>
> `transformers` 库会立刻：
>
> - 检查 `HF_ENDPOINT` 环境变量。
> - 如果你配置了，它就会从指定的镜像地址（如 `https://hf-mirror.com`）下载模型或数据。
> - 如果没有配置，它默认使用官方地址（`https://huggingface.co`），中国大陆用户往往访问困难或超时。
>
> ✅ 所以正确的顺序是：
>
> 必须在导入 `transformers` **之前**设置环境变量，比如这样：
>
> ```
> python复制编辑import os
> os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
> 
> from transformers import AutoTokenizer, AutoModel  # 必须放在后面
> ```
>
> 否则，`transformers` 模块加载时已经错过了读取环境变量的机会，配置就不会生效了。



## 简单入门

### 下载好的模型文件有哪些

我们以 `uer/gpt2-chinese-cluecorpussmall` 这个模型为例，观察以下其下载完成之后都有哪些东西。

在快照 `snapshots` 文件夹下有一些文件

1. `config.json`

   ```json
   {
     "activation_function": "gelu_new",
     "architectures": [
       "GPT2LMHeadModel"
     ],
     "attn_pdrop": 0.1,
     "embd_pdrop": 0.1,
     "gradient_checkpointing": false,
     "initializer_range": 0.02,
     "layer_norm_epsilon": 1e-05,
     "model_type": "gpt2",
     "n_ctx": 1024,
     "n_embd": 768,
     "n_head": 12,
     "n_inner": null,
     "n_layer": 12,
     "n_positions": 1024,
     "output_past": true,
     "resid_pdrop": 0.1,
     "task_specific_params": {
       "text-generation": {
         "do_sample": true,
         "max_length": 320
       }
     },
     "tokenizer_class": "BertTokenizer",
     "vocab_size": 21128
   }
   ```

   这是一个**自定义 GPT2 模型配置**，特点如下：

   - 模型结构和层数基本和 GPT2-base 保持一致（12层，768维，12头）；
   - 使用了 **BertTokenizer** 和 **中文词表大小（21128）** → 说明可能是 **用于中文的 GPT2**；
   - 启用了文本生成任务的典型设置（采样，320 长度）；
   - 关闭了 gradient checkpointing；
   - 配置合理，适用于 Hugging Face 的 `from_pretrained()` 或 `from_config()` 加载模型。

   我们可以重点关注一下下述这两个参数

   >   "tokenizer_class": "BertTokenizer", 表明使用了 BERT 分词器
   >   "vocab_size": 21128	词表大小为21128

2. vocab.txt

   里面记载了该模型能够识别的所有字（共21120个）

### 如何理解 `pipeline`

在 Hugging Face 的 `transformers` 库中，`pipeline` 是一个**高级封装工具**，它可以**快速调用各种预训练模型完成常见的 NLP 任务**，不用手动处理 tokenizer、模型加载、前处理、后处理等繁琐细节。

> `pipeline` 是 Hugging Face 提供的“傻瓜式一键推理接口”，**输入一句话，返回结果**，支持文本生成、情感分析、问答等任务。

下面是一个简易的加载模型并进行测试的场景代码

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

cache_dir = r"D:\data\model\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3"

model = AutoModelForCausalLM.from_pretrained(cache_dir)
tokenizer = AutoTokenizer.from_pretrained(cache_dir)


from transformers import pipeline

generator = pipeline(
    task = "text-generation",  # 注意是 "text-generation"，不是 "text-generate"
    model = model,
    tokenizer = tokenizer,
    device = 0  # device=0 表示使用第0号 GPU；如果用 CPU，可以省略这个参数
)

output = generator(
    "今天天气真好啊",  # 输入字符串直接写，不需要 `inputs=`
    max_length = 50,
    num_return_sequences = 1
)

print(output)

```

在不加调参的情况下，生成的结果是比较糟糕的。

* `AutoModelForCausalLM`是 Hugging Face Transformers 库中的一个**自动模型加载器类**，用于加载支持 **Causal Language Modeling（因果语言建模）** 的模型

  `AutoModelForCausalLM` 会根据你提供的模型配置（`config.json`），**自动选择合适的模型架构**，并加载预训练参数，**用于文本生成等任务**。 `AutoTkenizer` 同理，也是根据模型配置 `config.json` 来自动处理







### 其他注意事项

* 模型加载一般使用 `.from_pretrained()` 函数，且需要使用**绝对路径**，不然还会去 hugging face 中去下载

  目录是包含 `config.json` 的目录
  
* 所有在 transformers 中的数据都是以批次形式存在的，也就是说即便只有一个句子，它也是以 `[[sent]]` 这种形式存在而非 `[sent]`



## tokenizer_sentiment_analysis

###  重点理解 tokenizer 的过程以及重要的超参数

* 因为 tokenizer 和 model 是一一对应的（即 tokenizer 的输出是 model 的输入），所以一个使用的范式是：
  ```python
  batch_input = tokenizer(test_sequence,)
  model(**batch_input)
  ```

  * 输出：`{'input_ids': tensor, 'attention_mask': tensor}`

  * **工作过程**

    * `tokenizer`

      ```python
      test_senteces = ['today is not that bad', 'today is so bad', 'so good']
      batch_input = tokenizer(test_senteces, truncation = True, padding = True, return_tensors = "pt")
      batch_input
      ```

    * `tokenizer.tokenize()`

      这是一个分词函数，就是将输入的句子给分词，例如

      ```python
      tokenizer.tokenize(test_senteces[0],)
      ```

      ```bash
      ['today', 'is', 'not', 'that', 'bad']
      ```

    * `tokenizer.convert_tokens_to_ids()`

      用来把**一个或多个已分割好的 token（字符串）转换成对应的词表 ID（整数）**

      ```python
      tokenizer.convert_tokens_to_ids(tokenizer.tokenize(test_sentences[0],))
      ```

      ```bash
      [2651, 2003, 2025, 2008, 2919]
      ```

      没有开始 ID 序列和结束 ID 序列

    * `tokenizer.encode()`

      用于将 **一条文本（或一对文本）编码成 token 的 ID 序列**。

      它将输入的文本 **转换成 token ID 的列表**（整数）。返回的是 **纯整数列表**，不是字典，也没有 attention mask。

      ```python
      tokenizer.encode(test_senteces[0],)
      ```

      ```bash
      [101, 2651, 2003, 2025, 2008, 2919, 102]
      ```

      这里 101 和 102 分别是开始符和结束符；2651 对应 today，依次类推。

      可以发现 `tokenizer.encode()` ≈ `tokenizer.tokenize() + tokenizer.convert_tokens_to_ids()`

    * `tokenizer.decode()`

      用来将 **token ID 列表转换回可读文本字符串** 的方法。

      ```python
      tokenizer.decode([101, 2651, 2003, 2025, 2008, 2919,  102])
      ```

      ```bash
      '[CLS] today is not that bad [SEP]'
      ```

      `[CLS]` 和 `[SEP]` 是 **BERT** 以及很多基于 Transformer 的预训练模型中使用的特殊标记（special tokens），它们各自有特定的作用：

      | Token   | 作用                       |
      | ------- | -------------------------- |
      | `[CLS]` | 聚合序列信息，用于分类任务 |
      | `[SEP]` | 分割句子或文本片段         |

    **`tokenizer` 工作的原理其实就是 `tokenizer.vocab`：字典，存储了 token => id 的映射关系。当然这个字典中还包含了一些特殊的 `token`：`tokenizer.special_tokens_map`**

    **`tokenizer` 是服务于 `model` 的**

### 参数理解

以一个例子为例

```python
tokenizer(test_sentences, max_length = 32, truncation = True, padding = 'max_length', return_tensors = 'pt')
```

这里面 `max_length` 与 `padding` 是”不兼容“的。当你制定了 `max_length` 之后又想要填充，则需要令 `padding = 'max_length'`

或者

```python
tokenizer(test_sentences, truncation = True, padding = True, return_tensors = 'pt')
```

当我们把 `max_length` 消灭掉后，`padding` 就可以指定为 True 了，应该就是先遍历一遍所有的句子，然后将最长的那个句子的长度作为“`max_length`”，`padding = True` 即为填充

| 值             | 含义                                      | 说明                                 |
| -------------- | ----------------------------------------- | ------------------------------------ |
| `True`         | 启用 padding（默认对该 batch 内最长序列） | 动态 padding，适合模型效率优化       |
| `"longest"`    | 等价于 `True`                             | 只 pad 到当前 batch 中最长样本的长度 |
| `"max_length"` | 填充到 `max_length` 指定的固定长度        | 常用于模型训练（静态 shape）         |
| `False`        | 不进行 padding                            | 所有样本长度需一致，否则报错         |

事实上 `attention_mask` 也与 `padding` 相匹配。`attention_mask` 为 0 的部分即为 `padding` 的部分



### 输入到模型，理解模型的输出

```python
# 由于是推理的过程，我们不涉及训练，因此放入 no_grad()中
with torch.no_grad():
    outputs = model(**batch_input)
    print(outputs)
```

```bash
SequenceClassifierOutput(loss=None, logits=tensor([[-3.4620,  3.6118],
        [ 4.7508, -3.7899],
        [-4.1938,  4.5566]]), hidden_states=None, attentions=None)
```

可以将 logits 理解为：在送到 softmax 之前的输出。

我们可以对这个 logits 再次进行处理。见下：

```python
with torch.no_grad():
    outputs = model(**batch_input)
    print(outputs)
    scores = F.softmax(outputs.logits, dim = 1)
    print(scores)
    labels = torch.argmax(scores, dim = 1)
    print(labels)
```

```bash
SequenceClassifierOutput(loss=None, logits=tensor([[-3.4620,  3.6118],
        [ 4.7508, -3.7899],
        [-4.1938,  4.5566]]), hidden_states=None, attentions=None)
tensor([[8.4632e-04, 9.9915e-01],
        [9.9980e-01, 1.9531e-04],
        [1.5837e-04, 9.9984e-01]])
tensor([1, 0, 1])
```

可以看到输出的结果为 `[1, 0, 1]` 表示积极、消极、积极。

我们还可以查看模型的配置

```python
model.config
```

```bash
DistilBertConfig {
  "_name_or_path": "distilbert-base-uncased-finetuned-sst-2-english",
  "activation": "gelu",
  "architectures": [
    "DistilBertForSequenceClassification"
  ],
  "attention_dropout": 0.1,
  "dim": 768,
  "dropout": 0.1,
  "finetuning_task": "sst-2",
  "hidden_dim": 3072,
  "id2label": {
    "0": "NEGATIVE",
    "1": "POSITIVE"
  },
  "initializer_range": 0.02,
  "label2id": {
    "NEGATIVE": 0,
    "POSITIVE": 1
  },
  "max_position_embeddings": 512,
  "model_type": "distilbert",
  "n_heads": 12,
  "n_layers": 6,
  "output_past": true,
  "pad_token_id": 0,
  "qa_dropout": 0.1,
  "seq_classif_dropout": 0.2,
  "sinusoidal_pos_embds": false,
  "tie_weights_": true,
  "transformers_version": "4.30.2",
  "vocab_size": 30522
}
```

可以关注下其中的 `id2label`。我们再做一个映射

```python
with torch.no_grad():
    outputs = model(**batch_input)
    print(outputs)
    scores = F.softmax(outputs.logits, dim = 1)
    print(scores)
    labels = torch.argmax(scores, dim = 1)
    print(labels)
    # tensor的数据不能直接去做索引。要么 .item() 转换要么 .tolist() 
    labels = [model.config.id2label[id] for id in labels.tolist()]
    print(labels)
```

```bash
SequenceClassifierOutput(loss=None, logits=tensor([[-3.4620,  3.6118],
        [ 4.7508, -3.7899],
        [-4.1938,  4.5566]]), hidden_states=None, attentions=None)
tensor([[8.4632e-04, 9.9915e-01],
        [9.9980e-01, 1.9531e-04],
        [1.5837e-04, 9.9984e-01]])
tensor([1, 0, 1])
['POSITIVE', 'NEGATIVE', 'POSITIVE']
```



## tokenizer_encode_plus_token_type_ids

学习一下升级版的 `encode_plus`，它会得到 `token_type_ids`

初步理解一下这两个的区别（需要充分理解 input_ids, attention_mask, token_type_ids, padding 的含义与作用）：

| 功能                    | `encode`        | `encode_plus`         |
| ----------------------- | --------------- | --------------------- |
| 返回类型                | list            | dict                  |
| 只返回 input_ids        | ✅               | ❌（包含更多字段）     |
| 支持 attention_mask     | ❌               | ✅                     |
| 支持 token_type_ids     | ❌               | ✅                     |
| 支持 padding/truncation | ❌               | ✅                     |
| 用于模型输入            | ❌（需手动包装） | ✅（适合直接输入模型） |

### 复习

`bert-base-uncased` 是一个 **英语预训练 BERT 模型**，**12 层 Transformer 编码器结构**，输入是小写英文，**不区分大小写**（uncased）。可以根据自己输入的句子完成一些特定的下游任务

```python
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir = cache_dir)
tokenizer
```

自己探索一下该 `tokenizer` 的 special_token。将其编码之后再对其解码（很简单的任务）



### 认识文本语料（搁置，待补）

- `newsgroups_train.DESCR`
- `newsgroups_train.data`
- `newsgroups_train.target`
- `newsgroups_train.target_names`





## bert_model_architecture_params（bert 模型框架初探）

* 杂记

embeddings: BertEmbeddings

encode: BertEncoder: layer 0 ~ 11

pooler: BertPooler

要学会查看模型结构，看他的层，思考他在做什么

### 一些理解

Bert 是 transformer 的 encode 部分 （事实上还包括 embedding 部分和 pooler 部分），而 transformer 是一个 encoder - decoder（seq2seq）模型。

`BertForSequenceClassification` 就是一个 Bert 模型加了一个二分类的“头”，也就是我们所说的它是基于 Bert 构建的一个分类下游任务。

以上都是通过从接口层面直接观察模型结构得到的。

这个 Bert 模型只取了 transformer 的 encode 部分（也即 self-attention 和 feed-forward 部分）

> - `bert: encoder of transformer`
>   - `transformer: encoder-decoder(seq2seq)`
> - `bert`
>   - `embeddings`
>     - `word(token) embedding`
>     - `position embedding`
>     - `token type embedding`（其实就是 `segment embedding`）
>   - `encoder(12 layer)`
>     - `self attention (kqv)`
>     - `feed forward`
>   - `pooler`



```scss
输入文本 → 分词 → token_ids →
[嵌入层]
    └── Token Embeddings
    └── Segment Embeddings
    └── Position Embeddings
↓
[12 层 Transformer Encoder]
    每层：
        └── 多头自注意力（Multi-Head Self-Attention）
        └── Add & Norm
        └── 前馈网络（FeedForward Layer）
        └── Add & Norm
↓
[输出]
    └── 每个 token 的向量（Hidden State）
    └── [CLS] token 向量（句子表示）

```



* 关于 `CLS`

  `[CLS]` 是人为加的 —— 它的存在是 **为了提供一个专门用来表示“整句话”的向量**。

  > 原因在于：
  >
  > - Transformer 的输出是：**每个 token 的向量**
  > - 但很多任务（例如句子分类）**只需要一个向量代表整句话**
  > - 那用哪个 token 好呢？
  >
  > BERT 的设计者就说：
  >
  > > 那我们加一个专用 token，叫 `[CLS]`，让模型自动学会把“句子的意思”放在它里面。
  >
  > 在预训练时，BERT 有一个任务叫 NSP（Next Sentence Prediction）：
  >
  > > 模型要判断：两个句子是否连续？
  > >  —— 判断时就用 `[CLS]` 输出做分类。
  >
  > 因此，BERT 预训练过程中就**强迫模型学会了“把整句话的信息聚合到 `[CLS]` 上”**。

### 参数量统计

之所以叫大模型，是因为参数数量庞大。base 这个基础模型有12层，大约 1 亿个可训练参数。

**默认情况下，`pretrained BERT` 模型的所有参数都是可学习的（`requires_grad=True`）**，我们可以在下游任务中对整个模型进行微调（fine-tune）。但也可以选择只训练部分参数，这取决于你的任务需求和训练资源。

```python
total_params = 0
total_learnable_params = 0
total_embedding_params = 0
total_encoder_params = 0
total_pooler_params = 0

for name, param in model.named_parameters():
    print(name, "->", param.shape, "->", param.numel())

    if "embedding" in name:
        total_embedding_params += param.numel()
    
    if "encode" in name:
        total_encoder_params += param.numel()
    
    if "pooler" in name:
        total_pooler_params += param.numel()
    
    if param.requires_grad:
        total_learnable_params += param.numel()
    
    total_params += param.numel()
    
print(total_params)
print(total_learnable_params)
params = [total_embedding_params, total_encoder_params, total_pooler_params]
for param in params:
    print(param/sum(params))
```



```bash
embeddings.word_embeddings.weight -> torch.Size([30522, 768]) -> 23440896
embeddings.position_embeddings.weight -> torch.Size([512, 768]) -> 393216
embeddings.token_type_embeddings.weight -> torch.Size([2, 768]) -> 1536
embeddings.LayerNorm.weight -> torch.Size([768]) -> 768
embeddings.LayerNorm.bias -> torch.Size([768]) -> 768
encoder.layer.0.attention.self.query.weight -> torch.Size([768, 768]) -> 589824
# 局部

109482240
109482240
0.21772649152958506
0.776879099295009
0.005394409175405983
```

后续会提到何时会冻结一些 layer，何时不冻结任何 layer。



## no_grad_requires_grad

其实这是 Pytorch 中的知识，但是咱们可以再复习一下

- `torch.no_grad()`
  - 定义了一个上下文管理器，**隐式地**不进行梯度更新，**不会改变 `requires_grad`**（虽然在该环境下计算梯度不会更新，但是其仍然保留 `requires_grad` 的原本属性）
  - 适用于 `eval` 阶段，或 `model forward` 的过程中某些模块不更新梯度的模块（此时这些模块仅进行特征提取（前向计算），不反向更新）
- `param.requires_grad`
  - 显式地 `frozen` 掉一些 `module（layer）`的梯度更新
  - `layer/module` 级别
  - 可能会更**灵活**（微调的时候可以防止参数过多引起的显存爆炸）

### 验证

```python
def calc_learnable_params(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
    return total_params

print(calc_learnable_params(bert))	# 输出109482240

with torch.no_grad():
    print(calc_learnable_params(bert))	# 输出109482240，说明在该环境下不会该边模型参数 requires_grad 的属性
    
    
for name, param in bert.named_parameters():
    if param.requires_grad:
        param.requires_grad = False
print(calc_learnable_params(bert))		# frozen过程，输出 0
    
```



## bert_embedding-output

查看 bertmodel 的源码，我们发现其定义了 `embedding` , `encoder` , `pooler` 部分，这里介绍 `embedding` 部分的**前向过程**

这里的 `embeddings` 实际上是通过 `nn.embedding` + 索引查表实现的

### 小结

- `bert input embedding`：一种基于 `nn.embedding` + 索引的查表操作（lookup table）
  - 查表（这里的词典大小是 30522, `hidden_dim` 为 768）
    - `token embeddings`：30522 * 768
    - `segment embeddings`：2 * 768 ~~（它对整个批次是**共享**的，所以它这里不需要批次）~~
    - `position embeddings`: 512 * 768
  - 后处理
    - `layer norm`
    - `dropout`

### 相关源码以及理解前向过程

以下是 `BertEmbeddings` 的部分源码（我们只关注 `init` 部分），发现和小节上总结的一秩，我们有三个嵌入表 + 两个后处理构成

```python
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
```

其他需要注意的是

1. `embedding` 的呈现形式都是批的，因为接受的 `ids` 也是批的形式

2. 前向过程中，后处理之前得到的 `embedding` 是三个 `embedding` 相加，也就是

   ```python
   input_embed = token_embed + seg_embed + pos_embed	# [B, L, D]
   ```

   最终得到的 `embedding`  还要经过两个后处理过程



## subword_wordpiece_tokenizer

subword意为字词，wordpiece则是将一个词分片（将一个词拆分成多个字词）

首先我们需要明白，所有的词汇数量非常多，我们不可能将世界上所有的词汇都存入字典。`'bert-base-uncased'` 这个模型中存入的词汇数量为 30522，并没有覆盖所有的词。那么一定会有一些处理的手法。

### 词性简短分类

```python
s1 = 'albums sold 124443286539 copies'				# 数字型
s2 = 'technically perfect, melodically correct'		# melodically 将形容词转换成副词
s3 = 'featuring a previously unheard track'			# 不太常见的拼接前缀 unheard
s4 = 'best-selling music artist'						# 短横线形式
s5 = 's1 d1 o1 and o2'								# 子袋
s6 = 'asbofwheohwbeif'								# 无意义的字符
```

针对以上 6 种类型的“句子”，思考 tokenizer 该如何解决问题

将词拆分，分片处理



```python
tokenizer.vocab
tokenizer.ids_to_tokens
```

这两个一一映射



### 样本字词测试

需要明确的是，基于词汇表， `tokenize` `encode` `decode` 一体

1. `tokenize`：将 `word => token(s)`

   注意区分 `word` 和 `token(s)`，并且 `token(s)` 也是 `vocab` 中的 `keys`

2. `encode`：将 `token(s) => ids`

   通过查表的方式

3. decode：将 `ids => token => word`

* 针对数字

  ```python
  inputs = tokenizer(s1)
  print(inputs)
  print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))
  ```

  ```bash
  {'input_ids': [101, 4042, 2853, 13412, 22932, 16703, 20842, 22275, 2683, 4809, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
  ['[CLS]', 'albums', 'sold', '124', '##44', '##32', '##86', '##53', '##9', 'copies', '[SEP]']
  ```

  可见其先是将 `124443286539` 转换成 `'124', '##44', '##32', '##86', '##53', '##9'` 双井号表示拼接

* 副词类型

  `melodically => 'melodic', '##ally'`

* 前缀类型

  `unheard => 'un', '##heard'`

* 短横线形式（直接将短横线 - 给 split了，将一个 `word` 变成了 3 个 token）

  `best-selling => 'best', '-', 'selling'`

  事实上我们将 `bestselling` 这个分词会处理成 `'best', '##sell', '##ing'`, 

  > 我们建议将短横线 `-` 的这种处理方式规避掉，虽然前者和后者都将一个 word 转换成了 3 个token，但是从 token 角度的理解上`'best', '-', 'selling'` 可以看成**三个词**，而后者可以知道是一个词（##表拼接）

* 子袋

  `'s1 d1 o1 and o2' => '##1', 'd', '##1', 'o', '##1', 'and', 'o', '##2'`

* 摆烂字符

  `'asbofwheohwbeif' = > 'as', '##bo', '##f', '##w', '##he', '##oh', '##w', '##bei', '##f'`

感觉大概能摸索出规律了

### 小结

再次强调

- tokenizer **轻易不会**将一个词处理为 `[UNK] (100)`
- 基于词汇表，tokenize, encode, decode 一体
    - tokenize：word => token(s)，将word尽可能地映射为 vocab 中的 keys
    - encode: token => id
    - decode: id => token => word
        - encode 完了之后也不是终点（word），decode 还要能很好地将 id 还原，尽可能与输入的 word 对齐；





