const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const pty = require('node-pty');
const path = require('path');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

app.use(express.static(path.join(__dirname, 'public')));

wss.on('connection', (ws) => {
    console.log('Client connected');

    const shell = process.platform === 'win32' ? 'powershell.exe' : 'bash';
    
    const ptyProcess = pty.spawn(shell, [], {
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

    ws.on('message', (message) => {
        try {
            const msg = JSON.parse(message);
            if (msg.type === 'input') {
                ptyProcess.write(msg.data);
            } else if (msg.type === 'resize') {
                ptyProcess.resize(msg.cols, msg.rows);
            }
        } catch (e) {
            console.error('Message parse error:', e);
        }
    });

    ws.on('close', () => {
        console.log('Client disconnected');
        ptyProcess.kill();
    });

    ptyProcess.onExit(() => {
        console.log('PTY process exited');
        ws.close();
    });
});

const PORT = process.env.PORT || 3002;
server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
});
