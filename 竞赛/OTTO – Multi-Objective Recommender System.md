# OTTO – Multi-Objective Recommender System

## 记录

* 理解并学习 `.parquet` 这种数据

  * `.parquet` 的数据类型读取起来很快。

* 学会一些节约内存的 trick

  * 比如考虑将一些 `string` 类型的数据转化成类别 `int8` 极大地减少内存开销

  `.map()` 

  `.apply(lambda x: )`（这个很常用）

* 一些文件路径的处理

  * `glob.glob`：返回当前路径下所有**目标文件的路径**

* 进度条工具（让 GPT 给我情景，出题让我做）

  * `tqdm`，通过 `tqdm` 我们可以预估整段流程跑完的大致时间

* 时间戳工具（让 GPT 给我情景，出题让我做）

  * `datetime`
    * `datetime.fromtimestamp(x)`：将**时间戳形式**的 `x` 转换成日期 `datetime` 形式
    * `datetime.strptime()`
    * `.timestamp`：转换成时间戳
    * `.dt` ：是 **pandas 的 `.dt` 访问器**，用于访问 `datetime64` 类型列中的 **各类时间属性**，==就像 `.str` 是字符串**专用访问器**一样==。

* 其他一些小知识

  * 时间信息对于推荐任务还是比较重要的
    * 是工作日还是节假日？是早上还是晚上？这些都还是有一点影响的
  
* 一些用来加速的轮子

  * `df.groupby()` 分组操作（==很重要、很常用==）

    `df_train_offline.groupby(["aid", "type"], as_index = False).count()`

    是在对 `df_train_offline` 数据框按照 `"aid"` 和 `"type"` 这两列进行 **分组统计**，然后对**每组的==其他列==进行 `count()` 聚合**（即统计非空值的数量）。

    > 📌 `groupby(["aid", "type"], as_index=False)`
    >
    > - 将 `df_train_offline` 中具有相同 `(aid, type)` 值的记录分成一组。
    > - `as_index=False`：分组的结果会保留为普通列，而不是将 `aid` 和 `type` 设置为结果 DataFrame 的索引。
    >
    > 📌 `.count()`
    >
    > - 对每组中的所有列（除了 `aid` 和 `type`）执行非空值数量统计。
    > - 例如，如果原始 DataFrame 有 `session`, `ts`, `event` 等列，那么输出结果将包括这些列的计数。

    也就是说 `.count` 是统计这一组中的非空值，但是`.count()` 不会统计 `NaN` 值，跟 `.size()` 不一样

  * `groupby()` 与 聚合 `.agg()` 操作的配合使用
  
    考虑和 `as_index = False` 使用，这样操作更加简便。或者不使用 `as_index = False` 但是最后可以考虑加上 `.reset_index(drop = True)` `.reset_index()`这个东西比较厉害，能够重置`index`并将 `Series` 转换为 `DataFrame`（自己去想）
  
    单一聚合的时候可以不使用 `.agg` 。它更适用于 以下几种场景
  
    1. 多个聚合函数
  
       你想对同一列做多个聚合操作，比如 count 和 nunique：
  
       ```python
       df.groupby("type")["session"].agg(["count", "nunique"])
       
       # 这里表示只对 session 这一列操作
       ```
  
    2. 多个列，每列使用不同的聚合函数
  
       ```python
       df.groupby("type").agg({
           "session": "count",
           "aid": "nunique",
           "duration": "mean"
       })
       ```
  
    3. 自定义函数聚合
  
       若我们想传入 lambda 函数或自定义函数，则可以
  
       ```python
       df.groupby("type")["session"].agg(lambda x: x.max() - x.min())
       ```
  
       
  
  * `df.sort_values()`（==很重要、很常用==）


## 杂记

`df.iterrows()` 类似于链表中的头指针，事实上是一个迭代器，它比我们直接使用索引遍历更快，但是轮子会更快

可以使用 `%time` 来记录 `cell` 的运行时间



## 比赛背景

> 该比赛是一个**多目标推荐系统建模**竞赛，任务是基于用户在同一会话中的历史行为，**预测接下来的点击（clicks）、加购（cart additions）、和下单（orders）行为**。（**多目标推荐**）
>
> ------
>
> ### 🏁 比赛目标：
>
> 建立一个 **多任务模型（multi-objective model）**，同时预测：
>
> - 用户点击某商品的概率；
> - 用户将商品加入购物车的概率；
> - 用户最终下单的概率。
>
> ------
>
> ### 📌 比赛背景与意义：
>
> - 当前电商平台面临“选择过载”问题，用户在海量商品中容易迷失，导致放弃购买；
> - 更精准的推荐可以改善用户体验并提升转化率；
> - 虽然已有推荐系统（如矩阵分解、Transformer 模型等），但这些模型往往**只优化单一目标**；
> - 本比赛旨在推动构建**能同时优化点击率、加购率和转化率的推荐模型**。
>
> ------
>
> ### 🏢 数据来源：
>
> - 比赛数据来自德国最大在线零售商 OTTO（隶属于 Otto Group）；
> - 数据涵盖**1000万+ 商品、19000+ 品牌**；
> - 提供基于用户会话的行为序列数据（如点击、浏览、加购、下单等事件）。
>
> ------
>
> ### 🎯 最终目标：
>
> 参赛者提交一个模型或系统，能够根据用户在一个会话中的行为序列，预测用户接下来最可能进行哪类行为（点击、加购、下单），从而提升推荐的实时性与准确性。



## 载入数据与数据集划分



## 召回过程

* 使用了 `.groupby()`
* 使用了 `.sort_values()`



发挥自己的能动性，在充分理解赛题的情况下，自己搞点想法，做点实践
