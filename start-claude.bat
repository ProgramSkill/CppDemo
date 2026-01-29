@echo off
echo ========================================
echo 设置环境变量...
echo ========================================

set ANTHROPIC_AUTH_TOKEN=969ca57f48504699b6e0cf2fa0c5b5c4.U0kFQTZARbnfWyzn
set ANTHROPIC_BASE_URL=https://open.bigmodel.cn/api/anthropic
set API_TIMEOUT_MS=3000000
set CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

echo AUTH_TOKEN: %ANTHROPIC_AUTH_TOKEN:~0,20%...
echo BASE_URL: %ANTHROPIC_BASE_URL%
echo TIMEOUT: %API_TIMEOUT_MS% ms
echo ========================================
echo.
echo 正在启动 Claude Code...
echo.

claude %*