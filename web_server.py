#!/usr/bin/env python3
"""
Simple HTTP Server for the Voice Agent Test Web UI.
Serves the frontend and provides configuration endpoints.
"""

import asyncio
import json
import os
from aiohttp import web
from dotenv import load_dotenv

load_dotenv()

# Configuration
PORT = int(os.getenv("WEB_SERVER_PORT", 8000))
HOST = os.getenv("WEB_SERVER_HOST", "0.0.0.0")


# HTML Template
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Agent Test - Clean Tool Execution</title>
    <script src="https://unpkg.com/livekit-client/dist/livekit-client.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
        }

        h1 {
            text-align: center;
            margin-bottom: 10px;
            font-size: 1.8rem;
        }

        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
            font-size: 0.9rem;
        }

        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .card h2 {
            font-size: 1.1rem;
            margin-bottom: 15px;
            color: #4a9eff;
        }

        .form-group {
            margin-bottom: 15px;
        }

        label {
            display: block;
            margin-bottom: 5px;
            font-size: 0.85rem;
            color: #aaa;
        }

        input[type="text"],
        input[type="password"],
        input[type="url"],
        select {
            width: 100%;
            padding: 12px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: #fff;
            font-size: 0.95rem;
        }

        input:focus, select:focus {
            outline: none;
            border-color: #4a9eff;
        }

        .row {
            display: flex;
            gap: 15px;
        }

        .row > * {
            flex: 1;
        }

        .button-group {
            display: flex;
            gap: 15px;
            margin-top: 20px;
        }

        button {
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            flex: 1;
        }

        .btn-start {
            background: linear-gradient(135deg, #4a9eff 0%, #0066cc 100%);
            color: white;
        }

        .btn-start:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(74, 158, 255, 0.4);
        }

        .btn-stop {
            background: linear-gradient(135deg, #ff4757 0%, #cc0022 100%);
            color: white;
            font-size: 1.3rem;
            font-weight: bold;
            letter-spacing: 2px;
        }

        .btn-stop:hover:not(:disabled) {
            transform: scale(1.05);
            box-shadow: 0 5px 30px rgba(255, 71, 87, 0.5);
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* Status Indicator */
        .status-container {
            text-align: center;
            margin: 20px 0;
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 12px 24px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 50px;
            font-size: 1rem;
        }

        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #555;
            animation: pulse 2s infinite;
        }

        .status-dot.disconnected { background: #555; }
        .status-dot.connecting { background: #ffa500; }
        .status-dot.listening { background: #00ff88; }
        .status-dot.processing { background: #4a9eff; }
        .status-dot.speaking { background: #ff6b6b; }
        .status-dot.tool_executing { background: #ffd93d; }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        /* Conversation Log */
        .log-container {
            background: rgba(0, 0, 0, 0.4);
            border-radius: 8px;
            padding: 15px;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
        }

        .log-entry {
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .log-entry:last-child {
            border-bottom: none;
        }

        .log-time {
            color: #666;
            font-size: 0.75rem;
        }

        .log-user {
            color: #4a9eff;
        }

        .log-agent {
            color: #00ff88;
        }

        .log-system {
            color: #ffd93d;
        }

        .log-tool {
            color: #ff6b6b;
        }

        /* Tool Suggestions */
        .tool-suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }

        .suggestion-btn {
            padding: 8px 14px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .suggestion-btn:hover {
            background: rgba(74, 158, 255, 0.3);
            border-color: #4a9eff;
        }

        /* Debug Info */
        .debug-info {
            font-size: 0.8rem;
            color: #666;
            margin-top: 5px;
        }

        /* Scrollbar */
        .log-container::-webkit-scrollbar {
            width: 8px;
        }

        .log-container::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 4px;
        }

        .log-container::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Voice Agent Test</h1>
        <p class="subtitle">Clean Tool Execution Demonstration</p>

        <!-- Configuration Card -->
        <div class="card" id="config-card">
            <h2>🔧 Configuration</h2>
            <div class="row">
                <div class="form-group">
                    <label>LiveKit URL</label>
                    <input type="url" id="livekit-url" placeholder="wss://your-livekit-url.livekit.cloud" value="">
                </div>
                <div class="form-group">
                    <label>API Key</label>
                    <input type="text" id="api-key" placeholder="Your LiveKit API Key" value="">
                </div>
                <div class="form-group">
                    <label>API Secret</label>
                    <input type="password" id="api-secret" placeholder="Your LiveKit API Secret" value="">
                </div>
            </div>
            <div class="row">
                <div class="form-group">
                    <label>Room Name</label>
                    <input type="text" id="room-name" placeholder="voice-agent-test" value="voice-agent-test">
                </div>
                <div class="form-group">
                    <label>Participant Name</label>
                    <input type="text" id="participant-name" placeholder="User" value="User">
                </div>
            </div>
            <div class="row">
                <div class="form-group">
                    <label>LLM Provider</label>
                    <select id="llm-provider">
                        <option value="openai/gpt-4o-mini">OpenAI GPT-4o Mini</option>
                        <option value="openai/gpt-4o">OpenAI GPT-4o</option>
                        <option value="anthropic/claude-3-haiku">Claude 3 Haiku</option>
                        <option value="anthropic/claude-3.5-sonnet">Claude 3.5 Sonnet</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>STT Provider</label>
                    <select id="stt-provider">
                        <option value="deepgram/nova-2">Deepgram Nova-2</option>
                        <option value="deepgram/nova-3">Deepgram Nova-3</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>TTS Provider</label>
                    <select id="tts-provider">
                        <option value="cartesia/sonic-english">Cartesia Sonic</option>
                        <option value="elevenlabs/turbo-2">ElevenLabs Turbo-2</option>
                    </select>
                </div>
            </div>
            <div class="button-group">
                <button class="btn-start" id="start-btn" onclick="startSession()">🎤 Start Voice Session</button>
            </div>
            <div class="debug-info" id="debug-info"></div>
        </div>

        <!-- Status Card (hidden initially) -->
        <div class="card" id="status-card" style="display: none;">
            <h2>📊 Session Status</h2>
            <div class="status-container">
                <div class="status-indicator">
                    <span class="status-dot disconnected" id="status-dot"></span>
                    <span id="status-text">Disconnected</span>
                </div>
            </div>
            <div class="button-group">
                <button class="btn-stop" id="stop-btn" onclick="stopSession()">⏹️ STOP</button>
            </div>
        </div>

        <!-- Conversation Log -->
        <div class="card">
            <h2>💬 Conversation Log</h2>
            <div class="tool-suggestions">
                <button class="suggestion-btn" onclick="speak('What time is it?')">What time is it?</button>
                <button class="suggestion-btn" onclick="speak('Flip a coin')">Flip a coin</button>
                <button class="suggestion-btn" onclick="speak('Roll a 20 sided die')">Roll d20</button>
                <button class="suggestion-btn" onclick="speak('What\\'s the weather in Tokyo?')">Weather in Tokyo</button>
                <button class="suggestion-btn" onclick="speak('Calculate 234 times 567')">234 * 567</button>
                <button class="suggestion-btn" onclick="speak('Search for AI trends')">Search AI trends</button>
                <button class="suggestion-btn" onclick="speak('Analyze the sentiment of this text: I love this product')">Sentiment analysis</button>
            </div>
            <div class="log-container" id="log-container">
                <div class="log-entry">
                    <span class="log-system">System: Ready to start. Configure your settings above and click "Start Voice Session".</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        let room = null;
        let audioTrack = null;
        let isSessionActive = false;

        // Load saved config
        window.onload = function() {
            document.getElementById('livekit-url').value = localStorage.getItem('livekit-url') || 'wss://your-livekit-url.livekit.cloud';
            document.getElementById('api-key').value = localStorage.getItem('api-key') || '';
            document.getElementById('api-secret').value = localStorage.getItem('api-secret') || '';
            document.getElementById('room-name').value = localStorage.getItem('room-name') || 'voice-agent-test';
            document.getElementById('participant-name').value = localStorage.getItem('participant-name') || 'User';
            document.getElementById('llm-provider').value = localStorage.getItem('llm-provider') || 'openai/gpt-4o-mini';
            document.getElementById('stt-provider').value = localStorage.getItem('stt-provider') || 'deepgram/nova-2';
            document.getElementById('tts-provider').value = localStorage.getItem('tts-provider') || 'cartesia/sonic-english';
        };

        // Save config
        function saveConfig() {
            localStorage.setItem('livekit-url', document.getElementById('livekit-url').value);
            localStorage.setItem('api-key', document.getElementById('api-key').value);
            localStorage.setItem('api-secret', document.getElementById('api-secret').value);
            localStorage.setItem('room-name', document.getElementById('room-name').value);
            localStorage.setItem('participant-name', document.getElementById('participant-name').value);
            localStorage.setItem('llm-provider', document.getElementById('llm-provider').value);
            localStorage.setItem('stt-provider', document.getElementById('stt-provider').value);
            localStorage.setItem('tts-provider', document.getElementById('tts-provider').value);
        }

        // Log entry
        function addLog(type, message) {
            const container = document.getElementById('log-container');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            const time = new Date().toLocaleTimeString();
            entry.innerHTML = `<span class="log-time">[${time}]</span> <span class="log-${type}">${message}</span>`;
            container.appendChild(entry);
            container.scrollTop = container.scrollHeight;
        }

        // Update status
        function updateStatus(status, text) {
            const dot = document.getElementById('status-dot');
            dot.className = 'status-dot ' + status;
            document.getElementById('status-text').textContent = text;
        }

        // Connect to LiveKit Room (needs token from backend)
        async function getAccessToken(url, apiKey, apiSecret, roomName, participantName) {
            // For demo purposes, create token on the fly
            // In production, this should come from your backend
            try {
                const response = await fetch('/token', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url, apiKey, apiSecret, roomName, participantName })
                });
                if (!response.ok) throw new Error('Failed to get token');
                const data = await response.json();
                return data.token;
            } catch (e) {
                // Fallback: generate token on client side (NOT recommended for production)
                addLog('system', 'Error: Backend token endpoint not available. Please ensure the agent server is running.');
                throw e;
            }
        }

        // Start session
        async function startSession() {
            const url = document.getElementById('livekit-url').value;
            const apiKey = document.getElementById('api-key').value;
            const apiSecret = document.getElementById('api-secret').value;
            const roomName = document.getElementById('room-name').value;
            const participantName = document.getElementById('participant-name').value;

            if (!url || !apiKey || !apiSecret) {
                addLog('system', 'Error: Please fill in LiveKit URL, API Key, and API Secret');
                return;
            }

            saveConfig();
            document.getElementById('status-card').style.display = 'block';
            document.getElementById('start-btn').disabled = true;
            updateStatus('connecting', 'Connecting...');

            try {
                // Get access token
                const token = await getAccessToken(url, apiKey, apiSecret, roomName, participantName);

                // Connect to room
                room = new LivekitClient.Room();

                room.on(LivekitClient.RoomEvent.TrackSubscribed, (track) => {
                    if (track.kind === 'audio') {
                        attachAudioTrack(track);
                    }
                });

                room.on(LivekitClient.RoomEvent.ParticipantConnected, () => {
                    addLog('system', 'Agent connected to room');
                    updateStatus('listening', 'Listening');
                });

                room.on(LivekitClient.RoomEvent.DataReceived, (payload, participant) => {
                    const data = new TextDecoder().decode(payload);
                    try {
                        const parsed = JSON.parse(data);
                        if (parsed.type === 'status') {
                            updateStatus(parsed.status, parsed.text);
                        } else if (parsed.type === 'tool_call') {
                            addLog('tool', `Tool: ${parsed.tool}(${parsed.args}) - ${parsed.duration}ms`);
                        }
                    } catch (e) {}
                });

                await room.connect(url, token);
                isSessionActive = true;

                addLog('system', 'Connected to room: ' + roomName);
                addLog('system', 'Starting audio capture...');

                // Get user media
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioTrack = await LivekitClient.LocalAudioTrack.createAudioTrack('microphone', stream);
                await room.localParticipant.publishTrack(audioTrack);

                addLog('system', 'Session active. Start speaking!');

            } catch (e) {
                addLog('system', 'Error: ' + e.message);
                updateStatus('disconnected', 'Connection Failed');
                document.getElementById('start-btn').disabled = false;
            }
        }

        // Attach audio track
        function attachAudioTrack(track) {
            const element = track.attach();
            element.autoplay = true;
            document.body.appendChild(element);
            addLog('agent', 'Agent speaking...');
        }

        // Stop session
        async function stopSession() {
            if (room) {
                await room.disconnect();
                room = null;
            }
            if (audioTrack) {
                await audioTrack.stop();
                audioTrack = null;
            }
            isSessionActive = false;
            updateStatus('disconnected', 'Disconnected');
            document.getElementById('start-btn').disabled = false;
            addLog('system', 'Session ended');
        }

        // Speak something (for text-to-speech testing)
        function speak(text) {
            if (isSessionActive) {
                // This would send text to the agent via data channel
                room.localParticipant.publishData(
                    new TextEncoder().encode(JSON.stringify({ type: 'chat', text: text })),
                    LivekitClient.DataPacketKind.RELIABLE
                );
                addLog('user', text);
            } else {
                addLog('system', 'Please start the session first');
            }
        }
    </script>
</body>
</html>
"""


async def get_token(request: web.Request) -> web.Response:
    """Generate an access token for the LiveKit room."""
    try:
        data = await request.json()
        livekit_url = data.get("url")
        api_key = data.get("apiKey")
        api_secret = data.get("apiSecret")
        room_name = data.get("roomName", "voice-agent-test")
        participant_name = data.get("participantName", "User")

        # Import here to avoid issues if livekit is not installed
        from livekit import api

        # Create token
        token = api.AccessToken(api_key, api_secret) \
            .with_identity(participant_name) \
            .with_name(participant_name) \
            .with_grants(api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            ))

        return web.json_response({"token": token.to_jwt()})

    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


async def index(request: web.Request) -> web.Response:
    """Serve the main HTML page."""
    return web.Response(text=HTML_PAGE, content_type="text/html")


async def config(request: web.Request) -> web.Response:
    """Return configuration for the frontend."""
    config_data = {
        "livekitUrl": os.getenv("LIVEKIT_URL", ""),
        "defaultLlm": os.getenv("DEFAULT_LLM", "openai/gpt-4o-mini"),
        "defaultStt": os.getenv("DEFAULT_STT", "deepgram/nova-2"),
        "defaultTts": os.getenv("DEFAULT_TTS", "cartesia/sonic-english"),
    }
    return web.json_response(config_data)


async def health(request: web.Request) -> web.Response:
    """Health check endpoint."""
    return web.json_response({"status": "healthy"})


def create_app() -> web.Application:
    """Create and configure the web application."""
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/config", config)
    app.router.add_post("/token", get_token)
    app.router.add_get("/health", health)
    return app


async def main():
    """Run the web server."""
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, HOST, PORT)
    print(f"[WEB] Starting web server on http://{HOST}:{PORT}")
    await site.start()
    print(f"[WEB] Web UI available at http://localhost:{PORT}")

    # Keep server running
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        print("[WEB] Shutting down...")
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
