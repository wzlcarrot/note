### 数据类型

python的数据类型主要是int，float，string等类型

```python
a = 10
b = 10.1
c = "123"
print type(a)
print type(b)
print type(c)
```



### 数据类型转换

```python
a = "10"
b = "10.1"
c = 10

print int(a)
print float(b)
print str(c)
```

### 运算符

```python
# coding=utf-8
a = 2
b = 3

# 算术运算符
print a + b
print a - b
print a * b
print a / b
print a // b
print a ** b
```



### 字符串

这里主要是介绍字符串的拼接

```python
# coding=utf-8
a = "我"
b = "是刘德华"
print a+b
print "%s"%(a+b)
```

### 输入语句

```python
# coding=utf-8
a = input()     # a是数字类型
b = raw_input()   # b是字符串类型
```

### if语句

```python
a = 1
if a == 1:
    print "1"
elif a == 2:
    print "2"
else:
    print "3"
```



### while语句

```python
i = 0
while i < 100:
    print i
    i += 1
```



### for语句

```python
i = 0
for i in range(0, 100, 1):
    if i == 50:
        break
    print i
```



### 列表

#### 列表的定义语法

```python
# 定义一个空列表
l = []
l = list()
l = [1,2,3]
```

#### 列表的常用操作

```python
# coding=utf-8
l = [1,2,3,4]

# 在指定的下标位置插入元素
l.insert(1,6)

# 在列表的尾部插入一个新元素
l.append(100)

# 删除指定下标元素
del l[1]

# 删除列表的最后一个元素
l.pop()

# 删除指定元素的第一个匹配项
l.remove(1)

# 清空列表
l.clear()

# 统计某元素的数量
l.count(1)

# 计算列表的长度
len(l)
```



#### 列表的循环遍历

```python
l = [1, 2, 3, 4]
for t in l:
    print t

i = 0
for i in range(0, len(l), 1):
    print l[i]
    i += 1
```



### 元组

元组的元素不可修改。

#### 元组的定义

```python
# coding=utf-8
l = (1, 2, 3, 4)
l = ((1,2,3,4),(1,2,3,4))
# 定义空数组
l = ()
l = tuple()
```



#### 元组的操作

```python
# coding=utf-8
l = (1, 2, 3, 4)
# 通过下标取出内容
print l[0]

# 查找指定元素的下标
l.index(1)

# 统计某个元素的出现次数
l.count(1)

# 统计元组的长度
len(l)

# 元组的遍历
for t in l:
    print t
```



### 字符串

### 字符串的定义

```python
s = "123"
```



#### 字符串的操作

```python
# coding=utf-8
s = "12345"

# 查找指定元素的下标
value = s.index("1")

# 字符串.replace(字符串1,字符串2) 字符串2替换字符串1
s.replace("1","wu")

# split方法,返回值是列表
l = s.split(" ")

# 统计指定字符串出现的次数
cnt = s.count("1")

# 统计字符串的长度
length = len(s)
```

#### 字符串的遍历

```python
s = "12345"

for t in s:
    print s

i = 0
for i in range(0,len(s),1):
    print s[i]
```



### 数据容器切片

序列[起始下标:结束下标:步长]，如果步长为1，则可以写成 序列[起始下标:结束下标]。切片长度=结束下标-起始下标。

```python
s = [1, 2, 3, 4]
print len(s[1:3])
```

### 集合

集合中的元素都是唯一的。

#### 集合的定义

```
# coding=utf-8
s = {1,2,3,4,5}

# 定义空集合
s = {}
s = set()
```

#### 集合的操作

```python
# coding=utf-8
s = {1, 2, 3, 4, 5}

# 添加新元素
s.add(6)

# 移除指定元素
s.remove(1)

# 随机取出一个元素
element = s.pop()

# 清空集合
s.clear()

# 统计集合长度len()
len(s)
```



#### 集合的遍历

```python
# coding=utf-8
s = {1, 2, 3, 4, 5}

for t in s:
    print t
```



### 字典

字典为键值对的集合，字典的键必须是唯一的：

```python
a = {1:'a', 2:'b', 3:'c', 4:'d'}
print("a: ", a, type(a))
print("a[1]: ", a[1])

b = {'a':1, "bcd":2, 3.4:"e"}
print("b: ", b, type(b))
print("b['a']: ", b['a'])
```



字典的增删改查

```python
a = {'name': 'wjq', 'sex': 'male', 'qq': 1145141919}
print(a.get('name'))  # 通过键获取值，且不会报错
a.pop('qq')  # 删除键为 qq 的键值对
print(a.keys())  # 获取所有的键
print(a.values())  # 获取所有的值
a['name'] = 'lys'  # 改值
a['car'] = '宝马' #增
print("a: ", a, type(a))

if n in a.keys():
    print n
```



### 异常处理

#### 基本结构

利用 try，except，finally，else 语句处理异常：

```python
# -*- coding: UTF-8 -*-
try:
    a = 10 / 0
except ZeroDivisionError:
    # 如果出现了异常，则执行except这里的代码
    print "出现了异常"
else:
    # 如果没有出现异常，则会执行else这里的代码
    print "没有出现异常"
finally:
    # 不管出没有出现异常，都会执行这一块代码
    print "到此一游"
```



### 类

#### 基本用法

```python
# -*- coding: UTF-8 -*-
class Student:

    id = 1
    age = 18

    # self可以类比于java中的this指针
    def say_hi(self):
        print "id为%d的age为%d" % (self.id, self.age)


student = Student()
student.say_hi()

```



#### 构造方法

```python
# -*- coding: UTF-8 -*-
class Student:
    id = None
    name = None
    age = None

    def __init__(self, id, name, age):
        print "定义一个构造方法。"
        self.id = id
        self.name = name
        self.age = age

student = Student(1, "tom", 19)
id = student.id
name = student.name
age = student.age
print "id:%d,name:%s,age:%d" % (id, name, age)
```



#### 魔术方法

```python
# -*- coding: UTF-8 -*-
class Student:
    id = None
    name = None
    age = None

    def __init__(self, id, name, age):
        self.id = id
        self.name = name
        self.age = age

    # __str__魔术方法,类似于java中的toString()方法
    def __str__(self):
        return "id为%d,name为%s,age为%d" % (self.id, self.name, self.age)

    # __lt__魔术方法，
    def __lt__(self, other):
        return self.id < other.id

    # __le__魔术方法
    def __le__(self, other):
        return self.age <= other.age

    # __eq__魔术方法
    def __eq__(self, other):
        return self.age == other.age


stu1 = Student(1, "tom", 15)
stu2 = Student(2, "jack", 18)

print stu1
print stu1 < stu2

print stu1 >= stu2

print stu1 == stu2
```



#### 私有成员和方法

1. 成员变量和成员方法的命名均以__作为开头。

2. 类对象无法访问私有成员和方法
3. 类中其他成员可以访问私有成员和方法
4. 类可以通过访问公开成员和方法来访问私有成员和方法

##### 课后习题

```python
# coding=utf-8
class Phone:
    def __init__(self):
        pass

    __is_5g_enable = False

    def __check_5g(self):
        if self.__is_5g_enable:
            print "5g开启"
        else:
            print "5g关闭，使用4g网络"

    def call_by_5g(self):
        self.__check_5g()
        print "正在通话中"

phone = Phone()
phone.call_by_5g()
```



#### 继承

#### 单继承

和java中的extend继承非常相似

```python
# coding=utf-8
class Father:
    age = 50

    def __init__(self):
        pass

    def play(self):
        print "喜欢打扑克牌"


class Son(Father):

    def play(self):
        print "喜欢打游戏"


son = Son()
print son.age
son.play()
```



### 多继承

子类可以继承所有父类的属性和功能，并且可以重写。除此之外，在多继承中，如果父类有同名方法或者属性，先继承优先级高于后继承。

```python
# coding=utf-8
class Father:

    def __init__(self):
        pass

    def play1(self):
        print "喜欢打扑克牌"


class Mother:

    def __init__(self):
        pass

    def play2(self):
        print "喜欢打麻将"


class Son(Father, Mother):
    pass


son = Son()
print son.age
son.play1()
son.play2()

```















