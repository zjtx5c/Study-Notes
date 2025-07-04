## 学习路线

* 目前是跟着这个[教程](https://www.yuque.com/lianlianfengchen-cvvh2/zack/xdc15ohr77kwqglu)进行学习，打算先看文档，理解不透彻的话再去看视频。并在此处做一些重难点笔记。

### 变量

* 当我们想要取出数据00000010时，需要先访问地址00000101找到存储单元，然后取出存储单元存储的数据。（**即分两步骤**）再次理解以下，**存储单元**可以理解为一个**变量**，存储了数据00000010，变量的地址为00000101。

* 变量的声明与初始化：声明是声明，初始化是初始化

* 流对象：在 C++ 中，“**流对象（stream object）**”指的是用于**输入（input）和输出（output）操作**的一类对象，它们是通过类（如 `istream`、`ostream`）实现的，用来处理与**控制台、文件、字符串等介质之间的数据流动**。常见的流对象类别有标准输入输出流对象、文件流对象

  下面简介一下标准输入输出流对象：

  | 对象名 | 类别                | 说明                                 |
  | ------ | ------------------- | ------------------------------------ |
  | `cin`  | `istream`（输入流） | 用于从标准输入（通常是键盘）读取数据 |
  | `cout` | `ostream`（输出流） | 用于向标准输出（通常是屏幕）写入数据 |
  | `cerr` | `ostream`（错误流） | 用于输出错误信息，不带缓冲           |
  | `clog` | `ostream`（日志流） | 用于输出日志信息，带缓冲             |

  流对象是 C++ 中进行输入输出操作的核心组件。它们通过重载的 `<<` 和 `>>` 运算符，提供了**面向对象、可扩展的IO机制**，支持与多种数据源（终端、文件、内存）交互。

  | 运算符 | 方向 | 重载后的用途                   |
  | ------ | ---- | ------------------------------ |
  | `<<`   | 左移 | **输出**，把数据“送到”输出流里 |
  | `>>`   | 右移 | **输入**，从输入流中“提取”数据 |



### 类

* 其他的知识

  * 在 Windows 系统中，文件名通常是不区分大小写的，但在 Linux 或 macOS 中，`student.h` 和 `Student.h` 会被视为不同的文件。

  * 在类中的成员变量前面会多加一个下划线 `_`，目的是为了和参数做区分

  * `const` 修饰函数表示这个函数是一个常量函数，它不会修改成员变量的值（即它在这个函数里面不会去做任何修改这个函数属性的这些值（不会出现赋值操作），这是一种安全的声明）

    一般会对 `Get` 函数做这个修饰，你既然是获取，那你就不能修改。eg:

    ```cpp
    std::string GetName() const;	# 这是一个函数声明
    ```

  * `system(Command: "chcp 65001");`

  * 在不修改参数时，使用 `const` 类型引用可以减少拷贝（即减少栈的开销）

  * 在 C++ 中，**类的引用成员变量**必须通过 **初始化列表** 进行初始化，原因是 **引用必须在定义时被初始化**。引用不能被重新赋值，因此它的初始化必须在对象构造时完成。

    以下是详细原因：

    > **引用的初始化限制**：C++ 中的引用类型必须在声明时进行初始化，并且不能在后续的代码中更改为指向其他对象。引用不像指针，可以在构造函数中后续赋值。
    >
    > **初始化列表的作用**：类的构造函数中，可以通过初始化列表来初始化引用成员，确保它在构造函数体内之前就被正确地绑定到一个对象。如果在构造函数体内尝试对引用进行初始化或赋值，编译器会报错，因为引用不能在构造函数体中赋值或重新绑定。

  * 关于引用

    > **引用是绑定操作**：当你创建一个引用时，你并不是创建一个新的变量，而是为已有的变量提供一个别名。引用在初始化时绑定到一个对象，这个绑定过程就像是“锁定”了引用和对象的关系。之后，这个引用只能指向最初绑定的对象，无法再改变引用的目标。
    >
    > 例子：
    >
    > ```cpp
    > int a = 10;
    > int b = 20;
    > int& ref = a;  // 引用ref绑定到a
    > 
    > ref = b;  // 这行代码不会改变引用的绑定关系，ref仍然绑定到a，只是将a的值修改为20
    > std::cout << a << std::endl;  // 输出 20
    > std::cout << ref << std::endl;  // 输出 20
    > 
    > ```
    >
    > 在上述代码中，`ref` 初始时绑定到 `a`，这就意味着 `ref` 成为 `a` 的别名。
    >
    > 当执行 `ref = b;` 时，`ref` 并没有改变绑定，它依然绑定到 `a`，只是通过 `ref` 改变了 `a` 的值。
    >
    >
    > 其他注意点：
    >
    > **引用的绑定是不可变的**：一旦引用绑定到某个对象，你不能将它绑定到其他对象。它始终指向创建时指定的对象。
    >
    > **绑定时就确定了目标**：引用必须在声明时就进行绑定，你不能像指针一样在后续代码中改变引用的目标。
    >
    > **引用操作类似别名**：你可以认为引用是某个变量的“别名”，但这个“别名”在引用初始化后就不可更改。

  * `static` 是所有的对象所**共有**的成员。比如说班费就可以看作 `Student` 这个类中的 `static` 变量，因为班费不属于个人，属于整个集体

    它应该是放在对应类中的**共享段**区域

    在 C++ 中，**类的 `static` 成员变量**必须在类外进行初始化，通常放在类定义之外的源文件中。因为 `static` 成员变量是属于类的，而不是属于类的任何一个对象，它在所有对象之间共享，所以它的初始化需要在类的外部进行。

    > **静态成员是类的共享属性**：`static` 成员变量不属于任何特定的对象实例，而是属于类本身。它的生命周期跨越整个程序执行期间，因此它的初始化通常需要在类的外部进行。
    >
    > **类内声明，类外定义**：在类内，你通常只是**声明** `static` 成员变量，而不会为其分配内存或初始化它。实际的内存分配和初始化发生在类外。
    >
    > ```cpp
    > class MyClass {
    > public:
    >     static int staticVar;  // 仅声明，不初始化
    > };
    > 
    > // 在类外定义并初始化 static 成员变量
    > int MyClass::staticVar = 10;  // 这里进行初始化
    > ```
    >
    > 

* 理解类与对象

  * 对象是类的实例化
  * 汽车图纸（class）-> 汽车对象（object）
  * 对象是拥有类定义的所有属性和行为

* 封装、抽象、继承与多态

* 结构体里面所有的成员都是 public 的，外面任何一个方法或者任何一个作用域都能够访问这里面的变量成员，但是 class 不行，默认是 `private`

* 若类中不加访问权限的声明，默认所写的所有的属性都是私有的，但是 `struct` 都是公有的，都是可以被外部访问的。

  一般为了保证类的一个**封装效果**，我们习惯将类中的属性做成私有的，我们不希望被别人改（这在工程中是比较危险的）

  这样的话，外部不能直接改属性，但是我又想去修改这可怎么办？于是引出了一个概念：**成员函数**即方法，我们可以把成员函数写成公有的，通过访问成员函数去修改属性。

  成员函数一般在 `.h` 文件中是**声明**的，真正的修改是放在 `.cpp` 文件中实现的（习惯上）比如说是

  `Student.h` 与 `Student.cpp`

* **能不能有一种方式，在我定义一个对象的时候给他初始化呢**

  * 有的，**构造函数**，它是没有返回类型的。

* 构造函数

  * 默认构造（无参）
  
  * 有参构造（不管是无参构造还是有参构造，都是在**构造函数**中进行了初始化）
  
  * 参数初始化列表
  
  * 其他特殊情况的初始化
  
    1. 类的引用成员变量必须通过**初始化列表进行初始化**，因为引用必须在被定义/声明时初始化（不然引用变量的地址就指错了）
    2. `static` 类型的成员变量初始化不要放在构造函数中（具体见上文）
  
  * 拷贝构造（要用引用传参）深度理解一下（鸽了）
  
    关键在于**按值传递**会创建副本
  
    > 是的，我的意思是，在 **传值方式**（`Person(Person other)`）的拷贝构造函数中，编译器会 **自动调用拷贝构造函数** 来创建 `other` 对象的副本。这就是为什么会发生无限递归的原因。让我进一步解释这个过程，并对 **引用传递** 的情况进行说明。
    >
    > ### 1. **传值方式的递归调用**
    >
    > 考虑一下这个例子：
    >
    > ```cpp
    > class Person {
    > public:
    >     std::string name;
    > 
    >     // 错误的拷贝构造函数：传值方式
    >     Person(Person other) {  // 这里的other是传值
    >         name = other.name;
    >     }
    > };
    > 
    > int main() {
    >     Person p1("John");
    >     Person p2(p1);  // 这里调用拷贝构造函数
    > }
    > ```
    >
    > #### 发生的步骤：
    >
    > 1. **`Person p2(p1);`** 语句会触发 `Person` 类的拷贝构造函数。编译器会将 `p1` 作为参数传递给 `Person` 的构造函数。
    > 2. 然而，构造函数的参数是 **按值传递** (`Person other`)，这意味着：
    >    - 在拷贝构造函数内部，编译器会创建一个临时对象来存储 `p1` 的副本。
    >    - **为了创建这个副本，拷贝构造函数会再次被调用**，因为 `other` 是一个 `Person` 类型的对象，它需要通过拷贝构造函数来复制。
    > 3. **递归调用**：这个过程会不断重复，每次都创建一个新的副本，并再次调用拷贝构造函数，直到栈空间耗尽，导致栈溢出。
    >
    > 这就是为什么按值传递会引发 **无限递归** 的原因。
    >
    > ### 2. **引用传递的情况**
    >
    > 如果我们将拷贝构造函数的参数改为 **引用传递**（`Person&`），例如：
    >
    > ```cpp
    > class Person {
    > public:
    >     std::string name;
    > 
    >     // 错误的拷贝构造函数：传引用方式
    >     Person(Person& other) {  // 传引用，可能会引起递归
    >         name = other.name;
    >     }
    > };
    > 
    > int main() {
    >     Person p1("John");
    >     Person p2(p1);  // 这里会调用拷贝构造函数
    > }
    > ```
    >
    > #### 发生的步骤：
    >
    > 1. **`Person p2(p1);`** 会触发拷贝构造函数。
    > 2. **传递引用**：由于 `Person& other` 是一个引用传递，编译器**不会创建副本**，而是**直接传递** `p1` 的引用。
    > 3. 因此，**没有递归调用**，拷贝构造函数只会使用 `p1` 的引用来初始化 `p2`，而不会调用拷贝构造函数来创建副本。
    >
    > ### 3. **常量引用：避免递归的正确方式**
    >
    > 实际上，我们不应该使用 **非常量引用** 作为拷贝构造函数的参数，因为它仍然可能引发递归（特别是在复制过程中修改对象时），而应该使用 **常量引用**，即：
    >
    > ```cpp
    > class Person {
    > public:
    >     std::string name;
    > 
    >     // 正确的拷贝构造函数：传常量引用
    >     Person(const Person& other) {  // 使用常量引用
    >         name = other.name;
    >     }
    > };
    > 
    > int main() {
    >     Person p1("John");
    >     Person p2(p1);  // 正常调用拷贝构造函数，不会发生递归
    > }
    > ```
    >
    > #### 为什么使用 `const Person&` 能避免递归：
    >
    > - **常量引用**：`const Person& other` 使得 `other` 无法被修改，因此编译器不会尝试创建 `other` 的副本，也就避免了递归。
    > - 常量引用会直接使用对象的引用（即内存地址），而不需要创建副本。
    >
    > ### 4. **总结**
    >
    > - **按值传递**：当使用按值传递（`Person other`）时，编译器会自动创建 `other` 的副本，这样会导致 **递归调用** 拷贝构造函数，最终引发栈溢出。
    > - **按引用传递**：当使用 **引用**（`Person& other`）时，编译器不会创建副本，只是传递对象的引用，不会导致递归。
    > - **常量引用**：使用 **`const Person&`** 可以避免修改对象，同时避免递归调用拷贝构造函数，并且能正常执行。









## 语法

### 理解 `const`

`const` 表示这两个参数 **在函数体内不会被修改**。

在 C++ 中，传引用（`&`）虽然可以避免对象拷贝带来的开销（尤其是结构体、类等复杂类型），但是默认是可以修改引用指向的对象的。加上 `const`，就表示：

- 我只是**读取**这个参数，
- **绝对不会修改它的内容**。

#### 使用 `const` 的好处

**安全性提升**：

- 函数内部不小心写了 `a.max = 5;` 这样的代码，编译器会报错，避免误修改参数。

**语义更清晰**：

- 表示“这个函数只是用 `a` 和 `b` 来计算结果，不修改它们”。

**可以接受 `const` 参数**：

- ~~如果你传给这个函数的 `a` 或 `b` 是一个 `const Info` 类型的变量，那只有这个版本的函数才能接受，否则会报错。~~

**遵循 C++ 最佳实践**：

- 对于传引用且不会修改的参数，加 `const` 是**写高质量代码的基本习惯。**



### 结构化绑定

结构化绑定是 C++17 引入的一种语法糖，目的是让你可以“**同时声明多个变量**，并直接从 tuple、pair、结构体中**解包**”。

结构化绑定就是 C++ 的“解包”语法，让你能直接从 `pair/tuple/struct` 里取出多个值，不再写 `.first`、`.second`，但 **默认会拷贝，需要加 `&` 才是引用**！

* eg

不用结构化绑定

```c++
std::pair<int, std::string> p = {42, "hello"};
int a = p.first;
std::string b = p.second;
```

可以看出来非常啰嗦，如果使用结构化绑定，则

```c++
# 这里仅仅是拷贝
auto [a, b] = p;
```

如果想要“**引用原来的内容**”，就要这么写：

```c++
auto& [a, b] = p;
```

不会产生拷贝，适用于**大型对象、需要修改原数据等场景**（对象很大，且需要修改推荐使用）。

仅只读建议拷贝，但大型对象不建议拷贝，因为会占据内存；若要修改就一定要用引用了



###  关于对象的构造函数

* 构造函数名称与参数
* 初始化列表
  * 它用来初始化类成员变量

理解一下以下代码

```cpp
SegmentTree(int n_) : n(n_), tag(4 * n), info(4 * n) {}
```



### `T{}`

* 这是 C++11 引入的 **统一初始化语法（Uniform Initialization Syntax）**，`T{}` 表示用 **值初始化（value initialization）** 的方式构造一个 `T` 类型的对象。

在此之前我们使用了 `template <typename T>`

```c++
vector<int> a;
a.assign(5, int{});  // 就是 [0, 0, 0, 0, 0]

vector<string> b;
b.assign(3, string{});  // 就是 ["", "", ""]

vector<pair<int, double>> c;
c.assign(2, pair<int, double>{});  // 就是 [(0, 0.0), (0, 0.0)]

```

* `T{}` 会触发**值初始化**

  - 如果是内置类型（int、double 等）→ 变成 `0`

  - 如果是类/struct → 调用默认构造函数

* 能写 `T{}` 是因为 C++ 允许你使用 **花括号初始化一个类型的默认值**，它是 **构造一个默认值的合法方式**，语法非常泛化、强大、安全 —— 这是现代 C++ 的一个核心特性。



### 嵌套函数与辅助局部函数

因为 **C++ 不允许在函数内部直接定义另一个函数**（即 **嵌套函数**）。但 Python 或 JavaScript 可以。所以在写力扣时我发现不能在里面写普通函数。。

编译器会报错：

> `error: cannot declare a function within another function`

所以我们要使用局部辅助函数

1. 使用 lambda 表达式（推荐）

   - Lambda 可以访问外层函数的变量（通过 `[&]` 或 `[=]` 捕获）。
   - Lambda 是局部对象，不会污染外部作用域。

2. 改用类的成员函数

   如果辅助函数较复杂，可以将其定义为类的**私有成员**函数：

   ```cpp
   class Solution {
   private:
       int calc_bitnum(int st) {  // ✅ 类成员函数
           int cnt = 0;
           while (st) cnt++, st &= (st - 1);
           return cnt;
       }
   
   public:
       int maxProfit(...) {
           int bits = calc_bitnum(5); // 直接调用
           // ...
       }
   };
   ```

3. 使用 c++11 的 `std::function` （较少用）

   这个比较重（以后再补充）

## 容器

### vector

#### `.assign()`

`.assign` 是 C++ 标准库中 `std::vector` 的一个成员函数，用于**将指定值填充到向量中**，或者**将另一个容器的元素拷贝到当前向量中**。

* 将指定值填充到向量中

  ```cpp
  vec.assign(n, value)
  ```

  这里 `n` 是填充的元素个数，`value` 是填充到每个位置的值。

* 将另一个容器的元素拷贝到当前向量

  ```c++
  vec2.assign(vec1.begin(), vec1.end())
  ```



#### `.resize()`

`std::vector` 的 `.resize()` 是一个成员函数，用于**改变向量的大小**。它可以通过调整向量的大小来添加或删除元素。

* 改变向量大小

  ```c++
  vec.resize(new_size);
  ```

  **`new_size`**：是你想要改变的新的大小。

  如果新的大小比原来的大，`resize` 会通过默认构造值（例如对于 `int` 类型是 `0`）来扩展向量。

  如果新的大小比原来的小，向量的末尾元素会被删除。

* 指定新的大小和默认值

  ```c++
  vec.resize(new_size, value)
  ```

  **`new_size`**：新大小。

  **`value`**：新增元素的默认值。若 `new_size` 大于原来大小，则新增的元素都会被初始化为 `value`。



## 数值处理

### numeric

`<numeric>` 是 C++ 标准库中的头文件，专门包含**数值处理**相关的函数模板。虽然不像 `<algorithm>` 那么丰富，但它提供了一些很实用的工具，特别适合用于数值型容器（如 `std::vector`）的处理。

| 函数                                                    | 简介                                |
| ------------------------------------------------------- | ----------------------------------- |
| `std::accumulate`                                       | **求和/通用聚合**（可以自定义操作） |
| `std::reduce` (C++17)                                   | 类似 `accumulate`，但可能并行执行   |
| `std::inner_product`                                    | **内积/点积**（也可自定义）         |
| `std::adjacent_difference`                              | 相邻元素差值输出                    |
| `std::partial_sum`                                      | 前缀和计算（也可自定义操作）        |
| `std::exclusive_scan` / `std::inclusive_scan` (C++17)   | 高级前缀操作，适用于并行计算        |
| `std::transform_reduce` / `std::transform_scan` (C++17) | 结合 `transform` 和 `reduce/scan`   |

​	

#### `.accumulate()`

累加操作

```cpp
#include <numeric>
std::accumulate(v.begin(), v.end(), 0);
```



## 其他

### **默认构造行为**和**空容器访问**的细节

* `std::map<int, std::set<int>>` 的情况，**即对映射创建一个只有键而没有值是什么情况**

```c++
#include <bits/stdc++.h>

int main() {
    std::map<int, std::set<int>> st;
    st[1].insert(2);
    std::cout << st.size() << '\n';

    // 显示声明一个键，默认就是空
    st[2];
    std::cout << st.size() << '\n';

    for (auto& [k, v] : st) {
        std::cout << k * 3 << "--------" << '\n';
        std::cout << v.size() << '\n';
    }
    return 0;
}

/*
输出
1
2
3--------
1
6--------
0（这里的 0 不知道是什么意思，不管）
*/

```

可以发现，这种情况下，键能访问，但是值为空

以下是一些默认构造

| 类型             | 默认构造的值（`T{}` 或 `T()`） |
| ---------------- | ------------------------------ |
| `int`            | `0`                            |
| `double`         | `0.0`                          |
| `bool`           | `false`                        |
| `std::set<T>`    | **空**集合 `{}`                |
| `std::string`    | **空**字符串 `""`              |
| `std::vector<T>` | **空**向量 `[]`                |

针对这些**空**的情况，直接访问是比较危险的，并不建议这样做。
