# Debugging

调试技巧模块，涵盖断点、日志和性能分析。

## 📚 内容概览

| 主题 | 描述 | 难度 |
|------|------|------|
| 打印调试 | print语句调试 | ⭐ |
| 日志记录 | logging模块 | ⭐⭐ |
| 断点调试 | pdb调试器 | ⭐⭐⭐ |
| 性能分析 | cProfile, timeit | ⭐⭐⭐ |

## 🎯 学习目标

完成本模块后，你将能够：
- 使用日志记录程序运行状态
- 使用pdb设置断点调试
- 分析代码性能瓶颈
- 优化代码执行效率

## 💡 常用技巧

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def process_data(data):
    logger.debug(f"Processing: {data}")
    # ... processing logic
    logger.info("Processing complete")
```
