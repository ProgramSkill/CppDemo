# Object-Oriented Programming (OOP)

面向对象编程模块，涵盖类、继承和多态。

## 📚 内容概览

| 主题 | 描述 | 难度 |
|------|------|------|
| 类与对象 | 定义类、创建实例 | ⭐⭐ |
| 继承 | 父类、子类、方法重写 | ⭐⭐⭐ |
| 多态 | 接口统一、实现多样 | ⭐⭐⭐ |
| 封装 | 访问控制、属性装饰器 | ⭐⭐⭐ |

## 🎯 学习目标

完成本模块后，你将能够：
- 设计和实现类
- 使用继承复用代码
- 理解多态的概念和应用
- 正确封装类的内部实现

## 💡 核心概念

```python
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"
```
