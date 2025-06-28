# OTTO – Multi-Objective Recommender System

## 记录

* 理解并学习 `.parquet` 这种数据

  * `.parquet` 的数据类型读取起来很快。

* 学会一些节约内存的 trick

  * 比如考虑将一些 `string` 类型的数据转化成类别 `int8` 极大地减少内存开销

  `.map()` 

  `.apply(lambda x: )`（这个很常用）

* 一些文件路径的处理

  * `glob`

* 进度条工具

  * `tqdm`

* 时间戳工具

  * `datetime`
    * `datetime.fromtimestamp(x)`：将**时间戳形式**的 `x` 转换成日期 `datetime` 形式
    * `datetime.strptime()`
    * `.timestamp`：转换成时间戳
    * `.dt` ：是 **pandas 的 `.dt` 访问器**，用于访问 `datetime64` 类型列中的 **各类时间属性**，==就像 `.str` 是字符串**专用访问器**一样==。

* 其他一些小知识

  * 时间信息对于推荐任务还是比较重要的
    * 是工作日还是节假日？是早上还是晚上？这些都还是有一点影响的



## 载入数据与数据集划分

