# Chinese 项目系统架构文档

## 1. 项目概述

**项目名称：** Chinese Web API
**项目类型：** ASP.NET Core Web API
**架构风格：** Minimal API
**目标框架：** .NET 10.0

Chinese 是一个基于 ASP.NET Core 的轻量级 Web API 项目，采用 Minimal API 架构模式，提供 RESTful API 服务。项目集成了 OpenAPI 规范和 Scalar UI，提供交互式 API 文档界面。

## 2. 技术栈

### 2.1 核心框架
- **.NET 10.0** - 最新的 .NET 平台
- **ASP.NET Core** - Web 应用框架
- **Minimal API** - 轻量级 API 架构模式

### 2.2 主要依赖包
| 包名 | 版本 | 用途 |
|------|------|------|
| Microsoft.AspNetCore.OpenApi | 10.0.2 | OpenAPI 规范支持 |
| Scalar.AspNetCore | 2.12.15 | 交互式 API 文档 UI |

### 2.3 开发特性
- **Nullable 引用类型** - 启用空引用检查，提高代码安全性
- **隐式 Using** - 简化命名空间引用
- **Record 类型** - 不可变数据模型

## 3. 系统架构

### 3.1 架构模式
```
┌─────────────────────────────────────────┐
│          客户端 (HTTP Client)            │
└──────────────────┬──────────────────────┘
                   │ HTTP/HTTPS
                   ▼
┌─────────────────────────────────────────┐
│         ASP.NET Core 中间件管道          │
│  ┌────────────────────────────────────┐ │
│  │  HTTPS Redirection                 │ │
│  └────────────────────────────────────┘ │
│  ┌────────────────────────────────────┐ │
│  │  OpenAPI Middleware (Dev Only)     │ │
│  └────────────────────────────────────┘ │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│          Minimal API 端点路由            │
│  ┌────────────────────────────────────┐ │
│  │  GET /weatherforecast              │ │
│  └────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### 3.2 请求处理流程
1. **客户端请求** - 通过 HTTP/HTTPS 发送请求
2. **中间件处理** - 经过 ASP.NET Core 中间件管道
3. **路由匹配** - 根据 URL 路径匹配对应的端点
4. **业务逻辑** - 执行端点处理函数
5. **响应返回** - 返回 JSON 格式的响应数据

## 4. 核心组件

### 4.1 应用程序入口 (Program.cs)
**职责：** 应用程序的启动和配置中心

**主要功能：**
- 创建和配置 WebApplication 构建器
- 注册服务到依赖注入容器
- 配置中间件管道
- 定义 API 端点路由
- 启动应用程序

### 4.2 配置管理
**配置文件：**
- `appsettings.json` - 生产环境配置
- `appsettings.Development.json` - 开发环境配置

**配置项：**
- **Logging** - 日志级别配置
- **AllowedHosts** - 允许的主机头配置

### 4.3 数据模型
**WeatherForecast Record** (Program.cs:43)
```csharp
internal record WeatherForecast(DateOnly Date, int TemperatureC, string? Summary)
```

**特性：**
- 使用 `record` 类型，提供值相等性
- `internal` 访问级别，仅在当前程序集内可见
- 位置参数语法，简洁的属性定义
- 自动计算华氏温度属性

### 4.4 API 文档组件
**OpenAPI 集成：**
- 自动生成 OpenAPI 规范文档
- 访问路径：`/openapi/v1.json`

**Scalar UI：**
- 提供交互式 API 文档界面
- 访问路径：`/scalar/v1`
- 仅在开发环境启用

## 5. API 设计

### 5.1 API 端点列表

| 端点 | 方法 | 描述 | 响应格式 |
|------|------|------|----------|
| `/weatherforecast` | GET | 获取未来5天天气预报 | JSON Array |
| `/openapi/v1.json` | GET | OpenAPI 规范文档 | JSON |
| `/scalar/v1` | GET | Scalar API 文档界面 | HTML |

### 5.2 WeatherForecast API 详情

**端点名称：** GetWeatherForecast
**路径：** `/weatherforecast`
**方法：** GET
**认证：** 无

**响应示例：**
```json
[
  {
    "date": "2026-01-25",
    "temperatureC": 15,
    "summary": "Mild",
    "temperatureF": 59
  },
  {
    "date": "2026-01-26",
    "temperatureC": -5,
    "summary": "Freezing",
    "temperatureF": 23
  }
]
```

**响应字段说明：**
- `date` - 日期（ISO 8601 格式）
- `temperatureC` - 摄氏温度（-20 到 55 度）
- `summary` - 天气描述（10种预设描述之一）
- `temperatureF` - 华氏温度（自动计算）

## 6. 部署架构

### 6.1 运行环境要求
- **.NET 10.0 Runtime** - 必须安装 .NET 10.0 运行时
- **操作系统** - Windows、Linux 或 macOS
- **端口** - 默认使用 5055（HTTP）

### 6.2 部署模式

**开发环境：**
- 启用 HTTPS 重定向
- 启用 OpenAPI 端点
- 启用 Scalar UI 文档界面
- 详细日志记录

**生产环境：**
- 仅启用 HTTPS 重定向
- 禁用 OpenAPI 和 Scalar UI
- 精简日志记录（Warning 及以上级别）

### 6.3 配置管理
应用程序根据 `ASPNETCORE_ENVIRONMENT` 环境变量自动加载对应的配置文件：
- `Development` → `appsettings.Development.json`
- `Production` → `appsettings.json`

## 7. 开发指南

### 7.1 本地开发环境搭建
1. 安装 .NET 10.0 SDK
2. 克隆代码仓库
3. 进入 Chinese 项目目录
4. 运行 `dotnet restore` 恢复依赖包
5. 运行 `dotnet run` 启动应用程序

### 7.2 访问应用程序
- **API 端点：** http://localhost:5055/weatherforecast
- **OpenAPI 规范：** http://localhost:5055/openapi/v1.json
- **Scalar UI 文档：** http://localhost:5055/scalar/v1

### 7.3 测试 API
使用项目中的 `Chinese.http` 文件进行快速测试：
- 在 Visual Studio 或 VS Code 中打开 `Chinese.http`
- 点击请求上方的 "Send Request" 按钮
- 查看响应结果

### 7.4 项目结构
```
Chinese/
├── Program.cs                      # 应用程序入口和配置
├── Chinese.csproj                  # 项目文件
├── appsettings.json               # 生产环境配置
├── appsettings.Development.json   # 开发环境配置
├── Chinese.http                   # HTTP 请求测试文件
└── Properties/                    # 项目属性
```

### 7.5 添加新的 API 端点
在 `Program.cs` 中使用 Minimal API 语法添加新端点：

```csharp
app.MapGet("/your-endpoint", () =>
{
    // 业务逻辑
    return Results.Ok(new { message = "Hello" });
})
.WithName("YourEndpointName");
```

## 8. 技术特点与优势

### 8.1 Minimal API 优势
- **简洁性** - 减少样板代码，快速开发
- **性能** - 更少的抽象层，更高的性能
- **易维护** - 代码集中在单一文件中，易于理解

### 8.2 Record 类型优势
- **不可变性** - 线程安全，减少副作用
- **值相等** - 基于内容比较，符合数据模型语义
- **简洁语法** - 自动生成常用方法

### 8.3 OpenAPI 集成优势
- **自动文档** - 无需手动维护 API 文档
- **交互式测试** - Scalar UI 提供友好的测试界面
- **标准化** - 遵循 OpenAPI 规范，易于集成

## 9. 总结

Chinese 项目是一个现代化的 ASP.NET Core Web API 应用，采用 Minimal API 架构模式，具有以下特点：

- **轻量级架构** - 使用 Minimal API，代码简洁高效
- **现代化技术栈** - 基于 .NET 10.0 和最新的 C# 特性
- **完善的文档** - 集成 OpenAPI 和 Scalar UI
- **易于扩展** - 清晰的项目结构，便于添加新功能

该项目适合作为学习 ASP.NET Core Minimal API 的示例，也可以作为快速开发 Web API 的起点。

---

**文档版本：** 1.0
**最后更新：** 2026-01-24
**维护者：** Chinese 项目团队
