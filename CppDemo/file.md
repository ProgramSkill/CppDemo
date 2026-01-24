# GBK to UTF-8 Conversion Script

## 概述

此 PowerShell 脚本用于递归扫描当前目录及其子目录中的所有 `.cpp` 文件，并将 GBK 编码的文件转换为 UTF-8 编码。已经是 UTF-8 编码的文件将被跳过。

## 使用场景

- 将旧的 GBK 编码的 C++ 源文件转换为 UTF-8
- 统一项目中的文件编码格式
- 解决中文注释乱码问题

## 脚本代码
```powershell
Get-ChildItem -Path . -Recurse -Filter "*.cpp" | ForEach-Object { 
    $bytes = [System.IO.File]::ReadAllBytes($_.FullName); 
    $utf8String = [System.Text.Encoding]::UTF8.GetString($bytes); 
    $backToBytes = [System.Text.Encoding]::UTF8.GetBytes($utf8String); 
    if ([System.Linq.Enumerable]::SequenceEqual($bytes, $backToBytes)) { 
        Write-Host "Skipping UTF-8 file: $($_.FullName)" 
    } else { 
        $gbkString = [System.Text.Encoding]::GetEncoding("GB2312").GetString($bytes); 
        [System.IO.File]::WriteAllText($_.FullName, $gbkString, [System.Text.Encoding]::UTF8); 
        Write-Host "Converted GBK file: $($_.FullName)" 
    } 
}
```

## 使用方法

1. 打开 PowerShell
2. 导航到包含 `.cpp` 文件的目录
3. 复制并运行上述脚本

```powershell
# 示例
cd C:\Users\Csf\source\repos\CppDemo
# 粘贴并运行脚本
```

## 工作原理

1. 递归扫描所有 `.cpp` 文件
2. 读取文件字节内容
3. 检查是否已经是 UTF-8 编码
4. 如果是 GBK 编码，则转换为 UTF-8
5. 输出处理结果

## 注意事项

- ⚠️ 脚本会直接修改原文件，建议先备份
- ✅ 已经是 UTF-8 的文件会被跳过
- ✅ 脚本会输出每个文件的处理状态