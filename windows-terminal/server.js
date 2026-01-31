const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const pty = require('node-pty');
const path = require('path');
const fs = require('fs');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server});

app.use(express.static(path.join(__dirname, 'public')));

// 提供环境配置的API接口
app.get('/api/env-configs', (req, res) => {
    try {
        res.json(envConfigs);
    } catch (error) {
        console.error('Error serving env configs:', error);
        res.status(500).json({ error: 'Failed to load environment configurations' });
    }
});

const configPath = path.join(__dirname, 'config.json');
let envConfigs = {};

try {
    const configData = fs.readFileSync(configPath, 'utf8');
    envConfigs = JSON.parse(configData);
    console.log('Environment configurations loaded successfully');
} catch (error) {
    console.error('Failed to load config.json:', error);
    process.exit(1);
}

wss.on('connection', (ws) => {
    console.log('Client connected');
    
    let ptyProcess = null;
    let currentEnv = null;

    ws.on('message', (message) => {
        try {
            const msg = JSON.parse(message);
            
            if (msg.type === 'init') {
                // 初始化终端（不带环境变量）
                const shell = process.platform === 'win32' ? 'powershell.exe' : 'bash';
                
                ptyProcess = pty.spawn(shell, [], {
                    name: 'xterm-color',
                    cols: 120,
                    rows: 30,
                    cwd: process.env.HOME || process.env.USERPROFILE,
                    env: process.env
                });
                
                ptyProcess.onData((data) => {
                    try {
                        ws.send(JSON.stringify({ type: 'output', data }));
                    } catch (e) {
                        console.error('Send error:', e);
                    }
                });
                
                ptyProcess.onExit(() => {
                    console.log('PTY process exited');
                    ws.close();
                });
                
                console.log('Terminal initialized without environment variables');
                
            } else if (msg.type === 'switchEnv') {
                // 切换环境变量
                currentEnv = msg.env;
                const envConfig = envConfigs[currentEnv];
                
                if (!envConfig) {
                    console.error(`Environment configuration not found: ${currentEnv}`);
                    ws.send(JSON.stringify({ 
                        type: 'error', 
                        message: `Environment configuration not found: ${currentEnv}` 
                    }));
                    return;
                }
                
                // 在当前终端中设置环境变量
                const setEnvCommands = [
                    `$env:ANTHROPIC_AUTH_TOKEN="${envConfig.ANTHROPIC_AUTH_TOKEN}"`,
                    `$env:ANTHROPIC_BASE_URL="${envConfig.ANTHROPIC_BASE_URL}"`,
                    `$env:ENVIRONMENT_NAME="${envConfig.name}"`,
                    `Write-Host "Environment switched to: ${envConfig.name}" -ForegroundColor Green`
                ];
                
                setEnvCommands.forEach(command => {
                    ptyProcess.write(command + '\r\n');
                });
                
                console.log(`Environment switched to: ${currentEnv} (${envConfig.name})`);
                
                ws.send(JSON.stringify({ 
                    type: 'envSet', 
                    message: `Environment set to: ${envConfig.name}` 
                }));
                
            } else if (msg.type === 'input' && ptyProcess) {
                ptyProcess.write(msg.data);
            } else if (msg.type === 'resize' && ptyProcess) {
                ptyProcess.resize(msg.cols, msg.rows);
            }
        } catch (e) {
            console.error('Message parse error:', e);
        }
    });

    ws.on('close', () => {
        console.log('Client disconnected');
        if (ptyProcess) {
            ptyProcess.kill();
        }
    });
});

const PORT = process.env.PORT || 3002;
server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
