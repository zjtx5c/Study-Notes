## 一些内置函数与方法

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



### `.bit_length()`

在 **Python** 中，`int.bit_length()` 是一个非常实用的方法，用于返回一个整数的**二进制表示中所需的最小位数**（不包括符号位和前导零）。

* 例子见下：

```python
print((0).bit_length())   # 0
print((1).bit_length())   # 1 -> 0b1
print((5).bit_length())   # 3 -> 0b101
print((16).bit_length())  # 5 -> 0b10000
print((255).bit_length()) # 8 -> 0b11111111
```

负数行为

```python
print((-5).bit_length())  # 3 -> 0b101
```





## 数据结构（算法相关）

### `collections` 模块

Python 中的 `collections` 模块提供了许多额外的**容器数据类型**，这些类型可以帮助我们更高效地处理一些常见的数据结构任务。下面是 `collections` 模块中几个重要类的简要介绍：

#### `deque`

`deque`（双端队列）是 Python `collections` 模块中的一个类，它提供了一个高效的、双端操作的队列。与常规的队列（如`queue.Queue`）不同，`deque`可以在**两端快速地进行插入和删除**操作，**因此它在很多场景下比常规列表更高效**，尤其是当你需要频繁从队列两端添加或删除元素时。

* 常用方法

> **`append(x)`**：将元素 `x` 添加到右端。
>
> **`appendleft(x)`**：将元素 `x` 添加到左端。
>
> **`pop()`**：从右端删除并返回一个元素。
>
> **`popleft()`**：从左端删除并返回一个元素。
>
> **`extend(iterable)`**：将可迭代对象 `iterable` 中的元素添加到右端。
>
> **`extendleft(iterable)`**：将可迭代对象 `iterable` 中的元素添加到左端。
>
> **`rotate(n)`**：将队列元素旋转 `n` 位。若 `n` 为正数，右移；若为负数，左移。
>
> **`len(deque)`**：获取元素的数量
>
> 判断队列是否为空（1）使用 `len` （2）使用布尔值

#### `defaultdict`

`defaultdict` 是 `dict` 的一个子类，能够提供默认值。在正常的字典中，如果你访问一个不存在的键，会抛出 `KeyError` 异常。而 `defaultdict` 可以为不存在的键提供一个默认值，避免抛出异常。

==若针对的是 `set`，那么默认值是 `{}`，即空集==

```python
from collections import defaultdict
d = defaultdict(int)  # 默认值是 0
d['a'] += 1
d['b'] += 2
print(d)  # 输出: defaultdict(<class 'int'>, {'a': 1, 'b': 2})
```

也可以这样使用

```python
d = defaultdict(deque)
d = defaultdict(list)
```



### `sortedcontainers` 模块

`sortedcontainers` 库提供了几个常用且高效的数据结构，它们都与 ==**自动排序**== 相关。以下是一些主要的数据结构和它们的功能：

####  **`SortedList`**

- **作用**：提供一个自动保持有序的列表。
- **常用操作**：支持插入（`add`），删除（`remove`），索引访问（`[]`），切片（`[:]`），以及二分查找操作（`bisect_left`, `bisect_right`）。
- **时间复杂度**：插入和删除操作是 `O(log n)`，访问元素是 `O(1)`，切片操作是 `O(k)`（k 是切片的大小）。

> 适用于：需要频繁查询、删除和插入的场景，同时保证列表有序。且支持直接的索引下标操作，非常的方便。

示例：

```python
from sortedcontainers import SortedList
sl = SortedList([1, 2, 3])
sl.add(4)  # 插入元素
print(sl)  # [1, 2, 3, 4]
sl.remove(2)  # 删除元素
print(sl)  # [1, 3, 4]
```

#### **`SortedDict`**

- **作用**：提供一个自动保持有序的字典。
- **常用操作**：支持常规字典操作（如 `getitem`, `setitem`），但所有的键值对会自动保持键的有序性。
- **时间复杂度**：插入和删除操作是 `O(log n)`，查询和更新是 `O(log n)`。

> 适用于：需要字典的同时，还希望对键进行排序的场景。

示例：

```python
from sortedcontainers import SortedDict
sd = SortedDict({'a': 1, 'c': 3, 'b': 2})
print(sd)  # {'a': 1, 'b': 2, 'c': 3}
sd['d'] = 4
print(sd)  # {'a': 1, 'b': 2, 'c': 3, 'd': 4}
```

#### **`SortedSet`**

- **作用**：提供一个自动保持有序的集合（不允许重复元素）。
- **常用操作**：支持插入、删除和查询操作，自动去重，所有元素按升序排列。
- **时间复杂度**：插入、删除和查询操作都是 `O(log n)`。

> 适用于：需要有序集合且不允许重复元素的场景。

示例：

```python
from sortedcontainers import SortedSet
ss = SortedSet([3, 1, 2, 2])
print(ss)  # SortedSet([1, 2, 3])
ss.add(4)
print(ss)  # SortedSet([1, 2, 3, 4])
```

**特点：**

- 这些数据结构都基于**平衡树（如红黑树）**，提供高效的有序插入、删除和查询操作。
- 适用于需要频繁动态更新并保持排序的场景，比如在线排名、区间查询、滑动窗口等问题。
- 提供了比普通 `list` 或 `dict` 更高效的排序和查找操作，尤其适用于较大的数据集



## 算法

### `bisect`模块（当作二分用）

可以用来在已排序的列表中快速定位元素的插入位置。主要用于处理有序序列，以保持序列的有序性。

> **`bisect.bisect_left(a, x, lo=0, hi=len(a))`**
>
> - 返回 `x` 应该插入的位置，保持列表 `a` 的有序性。
> - 如果 `x` 已经存在，返回的是 `x` 的最左侧位置。
> - `lo` 和 `hi` 是可选的，表示查找的范围。
>
> **`bisect.bisect_right(a, x, lo=0, hi=len(a))`** 或 **`bisect.bisect(a, x, lo=0, hi=len(a))`**
>
> - 返回 `x` 应该插入的位置，保持列表 `a` 的有序性。
> - 如果 `x` 已经存在，返回的是 `x` 的最右侧位置（即插入位置位于 `x` 右边）。
>
> **`bisect.insort_left(a, x, lo=0, hi=len(a))`**
>
> - 向已排序的列表 `a` 中插入 `x`，保持有序，插入 `x` 到最左侧位置。
>
> **`bisect.insort_right(a, x, lo=0, hi=len(a))`** 或 **`bisect.insort(a, x, lo=0, hi=len(a))`**
>
> - 向已排序的列表 `a` 中插入 `x`，保持有序，插入 `x` 到最右侧位置。