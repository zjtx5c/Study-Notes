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

