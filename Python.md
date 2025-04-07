## 一些内置函数

### `fliter()`

> 在 Python 中，`filter()` 是一个用于从可迭代对象中过滤出符合特定条件的元素的内置函数。它会根据给定的函数筛选出可迭代对象中满足条件的元素，并返回一个==迭代器，而不是一个直接的可迭代对象==。
>
> ### 关于迭代器与可迭代对象的理解
>
> **迭代器 `iter()`**（`A`）本身是可迭代的，但它只允许按需（一个一个）地访问元素，并且只能遍历一次。
>
> ==如果你想反复访问或者多次操作元素，可以将 **迭代器** 转换为 **可迭代对象**==，比如通过 `list(A)` 来得到一个列表。
>
> 你可以对可迭代对象（例如列表）使用 `list()` 来将其转换为列表，但如果是迭代器，它的行为会在转换时“消耗”所有元素。
>
> ### 语法：
>
> ```python
> filter(function, iterable)
> ```
>
> - **`function`**: 一个函数，用于判断 `iterable` 中的每个元素是否符合某个条件。该函数接收一个元素作为输入，并返回 `True` 或 `False`。
>   - 如果返回 `True`，该元素就会被包含在最终的结果中。
>   - 如果返回 `False`，该元素会被排除。
>   - 如果 `function` 为 `None`，则会直接将每个元素与 `False` 进行比较（即排除掉所有“假”的元素，例如：`None`、`False`、`0`、空字符串等）。
> - **`iterable`**: 一个可迭代对象（如列表、元组、集合等）。
>
> ### 返回值：
>
> `filter()` 返回一个迭代器，包含符合条件的元素。你可以使用 `list()`、`tuple()` 或 `for` 循环等方式将其转换为列表或其他类型的数据结构。
>
> ### 示例 1：使用 `filter()` 过滤偶数
>
> ```python
> # 定义一个函数来判断一个数字是否为偶数
> def is_even(n):
>     return n % 2 == 0
> 
> # 一个数字列表
> numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
> 
> # 使用filter筛选偶数
> even_numbers = filter(is_even, numbers)
> 
> # 转换成列表并打印
> print(list(even_numbers))  # 输出: [2, 4, 6, 8, 10]
> ```
>
> ### 示例 2：使用 `filter()` 和 `lambda` 函数
>
> 你也可以使用 `lambda` 表达式来定义一个简单的条件函数。以下示例筛选出列表中的负数：
>
> ```python
> numbers = [10, -5, 3, -2, 7, -1]
> 
> # 使用lambda过滤负数
> negative_numbers = filter(lambda x: x < 0, numbers)
> 
> # 转换成列表并打印
> print(list(negative_numbers))  # 输出: [-5, -2, -1]
> ```
>
> ### 示例 3：`function` 为 `None` 时
>
> 如果将 `function` 设置为 `None`，`filter()` 会去掉所有被认为“假”的元素。
>
> ```python
> values = [0, 1, False, True, '', 'hello', None]
> 
> # 过滤掉所有假值（例如：0、False、空字符串、None等）
> truthy_values = filter(None, values)
> 
> # 转换成列表并打印
> print(list(truthy_values))  # 输出: [1, True, 'hello']
> ```
>
> ### 总结：
>
> - `filter()` 函数非常有用，可以用来从可迭代对象中过滤出符合某种条件的元素。
> - 它返回一个**迭代器**，你可以通过转换为其他类型（如列表）来查看结果。
> - 一般来说列表推导式可以完全替代 `filter()` 函数

### `open()`

> `open()` 返回的是一个 **文件对象（file object）**，用于读取或写入文件。
>
> ------
>
> ## **返回类型**
>
> **`open()` 返回的对象类型取决于 `mode`（打开模式）：**
>
> 1. **文本模式（`'r'`、`'w'`、`'a'` 等）**
>    - 返回 `TextIOWrapper` 类型的**文本文件对象**
>    - 读取内容时返回**字符串（`str`）**
> 2. **二进制模式（`'rb'`、`'wb'`、`'ab'` 等）**
>    - 返回 `BufferedReader` 或 `BufferedWriter` 类型的**二进制文件对象**
>    - 读取内容时返回**字节数据（`bytes`）**
>
> ------
>
> ## **示例**
>
> ### **1. 文本模式**
>
> ```python
> with open('example.txt', 'r', encoding='utf-8') as f:
>     print(type(f))  # <class '_io.TextIOWrapper'>
>     content = f.read()
>     print(type(content))  # <class 'str'>
> ```
>
> - **`f` 的类型**：`_io.TextIOWrapper`（文本文件对象）
> - **`content` 的类型**：`str`（字符串）
>
> ------
>
> ### **2. 二进制模式**
>
> ```python
> with open('example.jpg', 'rb') as f:
>     print(type(f))  # <class '_io.BufferedReader'>
>     content = f.read()
>     print(type(content))  # <class 'bytes'>
> ```
>
> - **`f` 的类型**：`_io.BufferedReader`（二进制文件对象）
> - **`content` 的类型**：`bytes`（字节流）
>
> ------
>
> ### **3. `pickle` 读取 `.pkl` 文件**
>
> ```python
> import pickle
> 
> with open('data.pkl', 'rb') as f:
>     print(type(f))  # <class '_io.BufferedReader'>
>     data = pickle.load(f)
>     print(type(data))  # 取决于 pkl 存的对象类型
> ```
>
> - **`f` 的类型**：`_io.BufferedReader`
> - **`data` 的类型**：取决于 `.pkl` 里存的对象，可能是 `dict`、`list`、`numpy.ndarray` 等
>
> ------
>
> ## **总结**
>
> | **模式**               | **返回的文件对象类型**                                       | **读取内容类型**  |
> | ---------------------- | ------------------------------------------------------------ | ----------------- |
> | `'r'`、`'w'`、`'a'`    | `_io.TextIOWrapper`（文本文件对象）                          | `str`（字符串）   |
> | `'rb'`、`'wb'`、`'ab'` | `_io.BufferedReader` / `_io.BufferedWriter`（二进制文件对象） | `bytes`（字节流） |
>
> ✅ **文本文件** → `str`
>  ✅ **二进制文件（如 `.pkl`、图片）** → `bytes`





### `map()`

语法： `map(function, iterable,...)`

返回对象：`map` 返回的是一个 **`map` 对象**，它是一个**迭代器**，而不是一个列表。为了查看它的内容，可以通过 `list()` 或 `for` 循环来将其转换为列表，或者直接迭代这个对象。一般情况下是能够支持解包操作的

`A, B, C = map(torch.tensor, (A, B, C))`

**`map(torch.tensor, (A, B, C))`**: 这部分代码会对 `(A, B, C)` 这个元组中的每个元素应用 `torch.tensor()` 函数。换句话说，它会将 `A`、`B` 和 `C` 分别转换成 PyTorch 的 tensor（张量）对象。`map` 会依次应用 `torch.tensor()` 函数到每个元素上。

**`A, B, C = ...`**: 这部分代码是对 **`map` 返回的结果进行解包**，将返回的张量分别赋值给 `A`、`B` 和 `C`。