# Windows Terminal with Environment Variable Support

这是一个支持多环境配置的 Windows Terminal 网页应用，允许用户在终端运行过程中动态切换不同的环境配置，并自动注入相应的环境变量。

## 功能特性

- 🌐 基于 Web 的终端界面
- ⚡ **即时启动** - 终端在页面加载时立即可用
- � **动态切换** - 运行过程中可随时切换环境配置
- �🔧 支持 3 组环境配置（开发、测试、生产）
- 🏷️ 终端标题显示当前选中的环境名称
- ⚙️ 自动注入 ANTHROPIC_AUTH_TOKEN 和 ANTHROPIC_BASE_URL 环境变量
- 📝 配置文件管理，便于维护

## 新的使用流程

1. **打开网页** - 终端立即显示并可用
2. **点击"切换"按钮** - 展开环境选择选项
3. **选择环境** - 点击所需的环境配置
4. **自动注入** - 环境变量立即注入到当前终端会话中
5. **随时切换** - 可以在终端运行过程中随时切换环境

## 安装和运行

1. 安装依赖：
```bash
npm install
```

2. 配置环境变量：
编辑 `config.json` 文件，填入你的实际环境变量值：
```json
{
  "development": {
    "name": "开发环境",
    "ANTHROPIC_AUTH_TOKEN": "your_dev_token_here",
    "ANTHROPIC_BASE_URL": "https://api.dev.anthropic.com"
  },
  "staging": {
    "name": "测试环境",
    "ANTHROPIC_AUTH_TOKEN": "your_staging_token_here",
    "ANTHROPIC_BASE_URL": "https://api.staging.anthropic.com"
  },
  "production": {
    "name": "生产环境",
    "ANTHROPIC_AUTH_TOKEN": "your_production_token_here",
    "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
  }
}
```

3. 启动服务器：
```bash
npm start
```

4. 打开浏览器访问：
```
http://localhost:3002
```

## 使用方法

### 基本使用
1. 打开网页后，终端立即可用
2. 点击环境配置区域的"切换"按钮展开选项
3. 点击选择所需的环境配置（开发环境/测试环境/生产环境）
4. 终端会自动注入对应的环境变量并显示确认信息
5. 终端标题会显示当前选中的环境名称

### 验证环境变量
在终端中使用以下命令验证环境变量：
```powershell
echo $env:ANTHROPIC_AUTH_TOKEN
echo $env:ANTHROPIC_BASE_URL
echo $env:ENVIRONMENT_NAME
```

### 动态切换
- 可以在终端运行过程中随时点击"切换"按钮来改变环境
- 新的环境变量会立即生效，无需重启终端
- 终端标题会实时更新显示当前环境

## 技术栈

- **前端**: HTML5, CSS3, JavaScript, xterm.js
- **后端**: Node.js, Express, WebSocket, node-pty
- **终端**: xterm.js + node-pty

## 项目结构

```
windows-terminal/
├── public/
│   └── index.html          # 前端界面
├── server.js               # 后端服务器
├── config.json             # 环境配置文件
├── package.json            # 项目依赖
└── README.md              # 说明文档
```

## 安全注意事项

- 请勿将包含真实 API 密钥的 `config.json` 文件提交到版本控制系统
- 建议使用环境变量或其他安全方式管理敏感信息
- 在生产环境中使用时，请确保适当的访问控制

## 自定义配置

你可以通过修改 `config.json` 文件来：
- 添加更多环境配置组
- 修改环境变量名称和值
- 自定义环境显示名称

## 故障排除

1. **终端无法启动**: 检查 `config.json` 文件格式是否正确
2. **环境变量未生效**: 确认配置文件中的变量名和值正确，检查终端中是否有错误信息
3. **连接失败**: 检查端口 3002 是否被占用
4. **切换环境无响应**: 刷新页面重新连接，或检查浏览器控制台是否有错误
