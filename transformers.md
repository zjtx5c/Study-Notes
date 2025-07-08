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



## 1_tokenizer_sentiment_analysis

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
| `False`        | 不进行 padding                            | **所有样本长度需一致，否则报错**     |

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



## 2_tokenizer_encode_plus_token_type_ids

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





## 3_bert_model_architecture_params（bert 模型框架初探）

* 杂记

embeddings: BertEmbeddings

encode: BertEncoder: layer 0 ~ 11

pooler: BertPooler

要学会查看模型结构，看他的层，思考他在做什么。

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



## 4_no_grad_requires_grad

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



## 5_bert_embedding-output

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



## 6_subword_wordpiece_tokenizer

subword意为字词，wordpiece则是将一个词分片（将一个词拆分成多个字词）

首先我们需要明白，所有的词汇数量非常多，我们不可能将世界上所有的词汇都存入字典。`'bert-base-uncased'` 这个模型中存入的词汇数量为 30522，并没有覆盖所有的词。那么一定会有一些处理的手法。

### 词性简短分类

```python
s1 = 'albums sold 124443286539 copies'				# 数字型
s2 = 'technically perfect, melodically correct'		# melodically 将形容词转换成副词
s3 = 'featuring a previously unheard track'			# 不太常见的拼接前缀 unheard
s4 = 'best-selling music artist'					# 短横线形式（尽量规避这种形式）
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



## 7_model_outputs

### 疑问

* 如何理解最后一层 `hidden` 的输出（`outputs[0]`）以及 `embedding` 的输出（`ouputs[2][0]`）

  > `outputs[0]` = `last_hidden_state`：最后一层 Transformer 输出的隐藏状态，融合了上下文信息。
  >
  > `outputs[2]` = `hidden_states` ：是一个元组，包含：
  >
  > - `hidden_states[0]`：embedding 层输出，也就是词向量+位置向量+segment向量加和的结果；
  > - `hidden_states[1]`：第1层 Transformer 的输出；
  > - ...
  > - `hidden_states[-1]`：最后一层 Transformer 的输出（同 `outputs[0]`）；
  >
  > 所以 `outputs[2][0]` 就是**embedding 层的输出**。**它是最初的向量**见上文 `embedding = token_embed + seg_embed + pos_embed`

* 如何理解该模型 `bert-base-uncased` 的 `forward` 过程？

  `embedding -> encode -> pooler`

### outputs

当我们在预加载模型（`from_pretrained` 阶段）进行设置 `output_hidden_states = True` 时。 `len(outputs)` 将会变为 3

```python
outputs = model(**token_input)
```

| 索引         | 内容名称            | 含义说明                                                     | 形状                                                         |
| ------------ | ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `outputs[0]` | `last_hidden_state` | 最后一层 Transformer 的输出（每个 token 的上下文表示）它是 token 粒度层面的 | `[batch_size, seq_len, hidden_size]` → `[1, 22, 768]`        |
| `outputs[1]` | `pooler_output`     | 对 `[CLS]` token 进行线性变换和激活的**句向量**（用于分类任务）它是**句子粒度层面**的。 | `[batch_size, hidden_size]` → `[1, 768]`                     |
| `outputs[2]` | `hidden_states`     | 包含 embedding 层 + 每一层 Transformer 的输出                | `13 × [batch_size, seq_len, hidden_size]` → `13 × [1, 22, 768]` |

进而我们有以下推测

1. `outputs[0] == outputs[2][-1]`

2. `outputs[1] == model.pooler(outputs[0])`

3. `outputs[2][0] == model.embeddings(token_input["input_ids"], token_input["token_type_ids"])`

4. ```python
   for i in range(len(outputs[2])):
       print(i, outputs[2][i].shape)
   ```

   ```bash
   0 torch.Size([1, 22, 768])
   1 torch.Size([1, 22, 768])
   2 torch.Size([1, 22, 768])
   3 torch.Size([1, 22, 768])
   4 torch.Size([1, 22, 768])
   5 torch.Size([1, 22, 768])
   6 torch.Size([1, 22, 768])
   7 torch.Size([1, 22, 768])
   8 torch.Size([1, 22, 768])
   9 torch.Size([1, 22, 768])
   10 torch.Size([1, 22, 768])
   11 torch.Size([1, 22, 768])
   12 torch.Size([1, 22, 768])
   ```

### 最后一层隐藏层进入 `pooler` 层时怎样工作的

即探究 `outputs[1] == model.pooler(outputs[0])` 

我们先找到具体的源码：

```python
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
```

`model.pooler(outputs[0])` 本质上做了这件事：

> BERT 中的 `pooler_output`（也就是 `outputs[1]`），**就是取 `[CLS]` 的词向量，做一个线性变换 + Tanh 激活后得到的结果**，被视为**整个句子的语义表示**（句向量 / sentence embedding）。

这也是 `outputs[1]` 的真实生成过程。

> 虽然这是 BERT 的原生句向量方案，但：
>
> - **它不一定效果最好**，因为 `[CLS]` 向量对句子的代表能力受训练目标影响；
> - 如果做检索、聚类、相似度任务，常用更强的句向量方法，比如：
>   - **平均池化（Mean Pooling）**：对 `last_hidden_state` 取平均；
>   - **Sentence-BERT**：用**专门训练过的句向量模型**，提升语义相似度对齐能力。

==我目前对这种做法还不是很能接受，随着后续地更深入了解应该会豁然开朗==



## 8_attn_01

我们会对比 model 的 self-attention 的输出以及自己写的计算 attention 的输出，进行对比看是否一样。目的是为了了解 attention 计算过程中的细节

### 疑问

* ```python
  from bertviz.transformers_neuron_view import BertModel
  from transformers import BertModel
  ```

  从这两个模块中导入的 BertModel 有什么区别？

  > 这两个 `BertModel` 来自不同的模块，功能和用途也有所不同：
  >
  > 1. **`from bertviz.transformers_neuron_view import BertModel`**：
  >    - 这是 `BertViz` 库中的 `BertModel`。`BertViz` 是一个可视化工具包，用于帮助研究人员和工程师分析和可视化 BERT 模型的注意力机制和内部特征。
  >    - 这个 `BertModel` 类是 `BertViz` 特定的扩展版本，提供了额外的可视化功能，如通过注意力得分来展示模型的工作原理。
  >    - 该模型主要用于神经网络层次、注意力分布等的可视化，不会改变模型的本质结构，但它有助于你理解模型在处理输入时的内部表现。
  > 2. **`from transformers import BertModel`**：
  >    - 这是 `transformers` 库中标准的 BERT 模型实现。`transformers` 是一个流行的自然语言处理（NLP）库，由 Hugging Face 开发，提供了许多预训练的模型（如 BERT、GPT、T5 等）。
  >    - 该 `BertModel` 实现的是 BERT 模型本身，用于进行常规的 NLP 任务（如文本分类、序列标注、问答等）。它没有任何专门的可视化功能，主要用于推理和训练。

* 为什么在做

  ```python
  emb_output[0] @ model.encoder.layer[0].attention.self.query.weight.T
  ```

  这个操作时要进行转置？
  因为 pytorch 的 linear 模块，在进行计算时是这样的： $y = xA^{\top} + b$ 。我们手动计算时是在复现 pytorch 的计算

### model config and load

```python
model.config
```

```python
{
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "finetuning_task": null,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "num_labels": 2,
  "output_attentions": true,
  "output_hidden_states": false,
  "pad_token_id": 0,
  "torchscript": false,
  "type_vocab_size": 2,
  "vocab_size": 30522
}
```

注意到这里的 `"hidden_size": 768`，其实是多个头“叠加”出来的结果。我们注意到这里的 `"num_attention_heads": 12` ，那么每一层每个头的隐层维度是 768 / 12 = 64。

我们可以取出来观察一下（从 `model.encoder` 这个模块中慢慢地取出来观察一下，**好好体会这个过程**！！！）

要会读懂这个架构（get）

```bash
BertEncoder(
  (layer): ModuleList(
    (0): BertLayer(
      (attention): BertAttention(
        (self): BertSelfAttention(
          (query): Linear(in_features=768, out_features=768, bias=True)
          (key): Linear(in_features=768, out_features=768, bias=True)
          (value): Linear(in_features=768, out_features=768, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (output): BertSelfOutput(
          (dense): Linear(in_features=768, out_features=768, bias=True)
          (LayerNorm): BertLayerNorm()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (intermediate): BertIntermediate(
        (dense): Linear(in_features=768, out_features=3072, bias=True)
      )
      (output): BertOutput(
        (dense): Linear(in_features=3072, out_features=768, bias=True)
        (LayerNorm): BertLayerNorm()
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
...
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
)
```



```python
model.encoder.layer[0].attention.self.query.weight.T[:, : 64]
```

```bash
tensor([[-0.0164, -0.0326,  0.0105,  ..., -0.0186, -0.0095,  0.0112],
        [ 0.0261,  0.0346,  0.0334,  ...,  0.0482, -0.0285, -0.0349],
        [-0.0263, -0.0423,  0.0109,  ..., -0.0724, -0.0453, -0.0304],
        ...,
        [ 0.0154, -0.0527, -0.0279,  ..., -0.0434,  0.0170,  0.0217],
        [ 0.0768,  0.1393,  0.0258,  ...,  0.0385,  0.0357, -0.0631],
        [ 0.0548,  0.0078, -0.0468,  ...,  0.0423, -0.0408,  0.0212]],
       grad_fn=<SliceBackward0>)
```

我们这样子索引其实是取出了**第一个头**的 `hidden_embedding`



### model output

```python
config = BertConfig.from_pretrained(model_name, output_attentions=True, 
                                    output_hidden_states=True, 
                                    return_dict=True,
                                    cache_dir = cache_dir)
config.max_position_embeddings = max_length

model = BertModel(config).from_pretrained(model_name, cache_dir = cache_dir)
```

我们进行这样的设置 `output_hidden_states=True, return_dict=True` 之后，我们的 `output` 会返回三个层，其中第三层和 `7_model_outputs` 这一节中讲得差不多，但是由于我们进行了 `return_dict=True` 这个操作

所以 `type(model_output[2]) == tuple` 并且 `type(model_output[2][0]) == dict`

当然 `model_output` 的第一层（0）和第二层（1）都是 `tensor` ，和之前第七章所讲的形式一样

```python
print(model_output[2][-1].keys())	#  return_dict=True
```

```bash
dict_keys(['attn', 'queries', 'keys'])
```



### from scratch

我们考虑计算第一层输出得到的 `attn` 系数

```python
# model_output[-1][0]['attn'].shape			# (B, H, N, N)
model_output[-1][0]['attn'][0, 0, :, :]		# (N, N)
```



```bash
tensor([[0.0053, 0.0109, 0.0052,  ..., 0.0039, 0.0036, 0.0144],
        [0.0086, 0.0041, 0.0125,  ..., 0.0045, 0.0041, 0.0071],
        [0.0051, 0.0043, 0.0046,  ..., 0.0043, 0.0045, 0.0031],
        ...,
        [0.0010, 0.0023, 0.0055,  ..., 0.0012, 0.0018, 0.0011],
        [0.0010, 0.0023, 0.0057,  ..., 0.0012, 0.0017, 0.0007],
        [0.0022, 0.0056, 0.0063,  ..., 0.0045, 0.0048, 0.0015]],
       grad_fn=<SliceBackward0>)
```





* my test

```python
import torch
import torch.nn.functional as F
# 这里注意取第一个头 [: 64]
Q_first_head_first_layer = torch.matmul(emb_output[0], model.encoder.layer[0].attention.self.query.weight.T[:, : 64]) + model.encoder.layer[0].attention.self.query.bias[: 64]
K_first_head_first_layer = torch.matmul(emb_output[0], model.encoder.layer[0].attention.self.key.weight.T[:, : 64]) + model.encoder.layer[0].attention.self.key.bias[: 64]
V_first_head_first_layer = torch.matmul(emb_output[0], model.encoder.layer[0].attention.self.value.weight.T[:, : 64]) + model.encoder.layer[0].attention.self.value.bias[: 64]

scores = torch.matmul(Q_first_head_first_layer, K_first_head_first_layer.T) / math.sqrt(64)
scores = F.softmax(scores, dim = -1)

print(scores)
print(torch.allclose(scores, model_output[-1][0]['attn'][0, 0, :, :]))		# (N, N)
```

```bash
tensor([[0.0053, 0.0109, 0.0052,  ..., 0.0039, 0.0036, 0.0144],
        [0.0086, 0.0041, 0.0125,  ..., 0.0045, 0.0041, 0.0071],
        [0.0051, 0.0043, 0.0046,  ..., 0.0043, 0.0045, 0.0031],
        ...,
        [0.0010, 0.0023, 0.0055,  ..., 0.0012, 0.0018, 0.0011],
        [0.0010, 0.0023, 0.0057,  ..., 0.0012, 0.0017, 0.0007],
        [0.0022, 0.0056, 0.0063,  ..., 0.0045, 0.0048, 0.0015]],
       grad_fn=<SoftmaxBackward0>)
True
```



## 9_BertSelfLayer 多头注意力机制（multi head attention）的分块矩阵实现

这个东西我在使用 dgl 实现 GAT 时已经仔细地理解过了。现在再来复习一下。

### 补充说明

1. 我们观察如下公式

   $W_q \in \mathbf{R}^{d_e \times d_q} \\W_k \in \mathbf{R}^{d_e \times d_k} \\W_v \in \mathbf{R}^{d_e \times d_v}$

   其中 $d_e$ 是输入层的特征维度

   根据 transformers 的公式，我们要求 $d_q = d_k$，但是对 $d_v$ 的大小不做具体要求，但是我们有时候为了将维度统一贯穿整个过程，一般这三个维度的大小是相同的

2. 自己手写一遍 multi-layer 的分块矩阵的实现

   **需要注意的是，多头的实现是 softmax 作用在单头上再拼接而不是先拼接再 softmax**



## 10_add_norm_residual_conn

和上面一样，学会自己实践一下

### 疑问

* `model.config` 中的 `intermediate_size` 的含义是什么？如何理解它的对齐。

  就是前馈神经网络中拉高的维度，在 bert-base-uncased 中是 768 * 4 = 3072

* 理解 `torch.no_grad()`  和 `model.eval()` 的区别

  > ### `torch.no_grad()`
  >
  > - **目的**：禁用梯度计算。
  > - **作用**：在该上下文环境下，所有的张量操作都不会计算梯度，从而节省显存并加速推理过程。它常用于 **推理（inference）** 阶段，因为在此阶段你不需要计算梯度。
  > - **使用场景**：例如，推理或验证模型时，因为你不打算进行反向传播和梯度更新。
  >
  > ```python
  > 辑with torch.no_grad():
  >     output = model(input)  # 推理时不计算梯度
  > ```
  >
  > ### `model.eval()`
  >
  > - **目的**：将模型设置为评估模式。
  > - **作用**：它主要影响某些特定层，如 **BatchNorm** 和 **Dropout** 层。
  >   - **BatchNorm**：在训练模式下，BatchNorm 会使用当前批次的均值和方差进行归一化，而在评估模式下，它会使用训练过程中计算得到的全局均值和方差。
  >   - **Dropout**：在训练模式下，Dropout 会随机丢弃一些神经元，以防止过拟合；而在评估模式下，Dropout 会被禁用，即每个神经元都会被保留，以便在推理时使用完整的模型。
  >
  > ```python
  > model.eval()  # 进入评估模式
  > output = model(input)  # 评估阶段
  > ```

  



### BertLayer层

以 `layer[0]` 为例（其他都是一样的）

```bash
(layer): ModuleList(
      (0): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
```

和最经典的那张图对应起来（仔细思考一下）

- BertLayer
    - attention: BertAttention
        - self: BertSelfAttention
        - output: BertSelfOutput
    - intermediate: BertIntermediate, 768=>4*768
    - output: BertOutput, 4*768 => 768

### 自己复现一下

已 get

注意：在 `mha_output = layer.attention.self(embeddings)` 在这一界面，即在 attention 层计算自注意力机制输出的会是一个 `tuple` ，第一项 `mha_output[0]` 存入的是 Multi-Head Attention 。（`[B, seq_len, heads * heads_hidden]`）



## 11_bert_head_pooler_output

### 理解 pooler_output（流程）

* `pooler_output` 是最终的一个输出
* 输入的东西是什么？
* 输出的东西是什么？
* 具体是怎么计算的？
  * 线性层 + 激活层



### **forward and pooler output**

```python
bert.eval()
with torch.no_grad():
    output = bert(**inputs)
    # output.keys() odict_keys(['last_hidden_state', 'pooler_output'])
    print(output["pooler_output"].shape)
```

```bash
torch.Size([1, 768])	# (B, D)
```

**pool_output 的做法是选择每个一句子（B）中的首个 token 的 embedding （in_D = D），将其送入一个线性层 + 激活层，最终得到一个代表整个句子的输出（B,  out_D = D）**

这样我们才可以使用这个最终的输出做一些句子分类、情感分析的任务，**将落点落到句子上**。

我们可以进行验证

### **from scratch**

```python
bert.pooler
```

```bash
BertPooler(
  (dense): Linear(in_features=768, out_features=768, bias=True)
  (activation): Tanh()
)
```

```python
my_output = bert.pooler.activation(bert.pooler.dense(output["last_hidden_state"][0, 0, :]))
print(my_output.shape)
torch.equal(my_output, output["pooler_output"][0])
```

```bash
torch.Size([768])
True
```

### bert Head

Bert 这个架构，它的 embedding 和 encoder 是不会变的，这也是它的精华所在。它后面跟了不同的 head，意为着它接了不同的任务，我们只需要改后面的 head 即可。

Bertmodel 的默认 head 是 pooler。

```bash
  (pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
```





## 12_masked_lm

这是很经典的语言模型：masked_language_model：掩码语言模型

### 小节

* 理解 mlm 任务中 BertOnlyMLMHead 的组织结构和工作流程
  * decoder 是将 768 映射到词汇表 30522 上
* 对多分类任务过程（下游任务）和网络结构（`mlm.cls`）、构造 `labels` 与 `input_ids` 过程（即 `masking` 过程）、计算损失过程、翻译过程更加清晰透彻了。

### mlm

```python
mlm = BertForMaskedLM.from_pretrained(model_name, output_hidden_states=True, cache_dir = cache_dir)
mlm
```

```bash
# 前面的省略和之前老生常谈的 embedding 和 encoder 一样，只看 head 层
(cls): BertOnlyMLMHead(
    (predictions): BertLMPredictionHead(
      (transform): BertPredictionHeadTransform(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (transform_act_fn): GELUActivation()
        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      )
      (decoder): Linear(in_features=768, out_features=30522, bias=True)
    )
  )
```

从 `in_features` 和 `out_feats` 可以看出，这是一个针对对应此表的多分类任务。

解码器就相当于输出层了，这个网络也很好理解。



### masking 过程

注意：

* 开始 token 与 结束 token 不要设置成掩码，这是约定俗称的。



学会生成 `inputs["labels"]`

```python
# 以其中的第一个句子为例
inputs["labels"] = inputs["input_ids"].detach().clone()
mask = torch.rand(inputs["labels"].shape) < 0.15
mask_arr = (mask) * (inputs["labels"] != 101) * (inputs["labels"] != 102)	# 不能将开始和结束符设置为掩码
selection = torch.flatten(mask_arr[0].nonzero()).tolist()	# 取第一个句子
inputs["input_ids"][0][selection] = 103
inputs
```

```bash
{'input_ids': tensor([[  101,  2044,  8181,  5367,  2180,  1996,  2281,  7313,  4883,  2602,
          2006,   103,  3424,  1011,   103,  4132,  1010,  2019,  3988,  2698,
          6658,  2163,  4161,  2037, 22965,  2013,  1996,  2406,   103,  2433,
          1996, 18179,  1012,  2162,  3631,   103,  1999,  2258,  6863,  2043,
         22965,  2923,  2749,  4457,  3481,  7680,  3334,  1999,  2148,  3792,
          1010,  2074,  2058,  1037,   103,  2044,  5367,   103,  1055, 17331,
           103,   102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), 'labels': tensor([[  101,  2044,  8181,  5367,  2180,  1996,  2281,  7313,  4883,  2602,
          2006,  2019,  3424,  1011,  8864,  4132,  1010,  2019,  3988,  2698,
          6658,  2163,  4161,  2037, 22965,  2013,  1996,  2406,  2000,  2433,
          1996, 18179,  1012,  2162,  3631,  2041,  1999,  2258,  6863,  2043,
         22965,  2923,  2749,  4457,  3481,  7680,  3334,  1999,  2148,  3792,
          1010,  2074,  2058,  1037,  3204,  2044,  5367,  1005,  1055, 17331,
          1012,   102]])}
```



### forward and calcuate loss

```python
mlm.eval()
with torch.no_grad():
    output = mlm(**inputs)
output.keys()
```

```bash
odict_keys(['loss', 'logits', 'hidden_states'])
```

`output["logits"]` 其实就是还没有经过 softmax 的最终输出，我们可以进行验证



```python
tmp = mlm.cls(output["hidden_states"][-1])
torch.equal(tmp, output["logits"])
```

```bash
True
```



或者我们再细化一下

```python
mlm.eval()
with torch.no_grad():
    transformed = mlm.cls.predictions.transform(last_hidden_state)
    print(transformed.shape)
    logits = mlm.cls.predictions.decoder(transformed)
    print(logits.shape)
logits == output["logits"]
```

```bash
tensor([[[True, True, True,  ..., True, True, True],
         [True, True, True,  ..., True, True, True],
         [True, True, True,  ..., True, True, True],
         ...,
         [True, True, True,  ..., True, True, True],
         [True, True, True,  ..., True, True, True],
         [True, True, True,  ..., True, True, True]]])
```



### **loss and translate**

要学会这个过程！！

1. 对交叉熵的理解加深了
2. 会计算 loss 和翻译了



具体过程去看 `10_masked_lm.ipynb` 这个文件
