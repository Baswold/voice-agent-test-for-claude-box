# Voice Agent Test Application

A simple voice agent application to test **clean tool execution** with varying complexity levels. Built with LiveKit Agents (Python).

## Features

- **10 Test Tools** with varying execution times:
  - Quick tools (< 1s): `get_current_time`, `flip_coin`, `roll_dice`
  - Medium tools (1-3s): `get_weather`, `calculator`, `web_search`
  - Long tools (3-10s): `analyze_text`, `generate_report`, `database_query`
  - Edge case: `trigger_error` for testing error handling

- **Silent Tool Execution**: Tools execute WITHOUT the agent announcing them
- **Web UI** with:
  - API key configuration
  - Provider selection (STT/TTS/LLM)
  - Large RED STOP button
  - Real-time status indicator
  - Conversation log

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/Baswold/voice-agent-test-for-claude-box.git
cd voice-agent-test-for-claude-box

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\\Scripts\\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Required: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
# Plus: OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### 3. Run

```bash
# Start the agent server
python voice_agent.py

# In another terminal, start the web UI
python web_server.py

# Open browser to http://localhost:8000
```

## Testing

### Console Mode (Local Testing)

```bash
# Run in console mode for quick local testing
python voice_agent.py console
```

### Web UI Testing

1. Open http://localhost:8000
2. Enter your LiveKit credentials
3. Click "Start Voice Session"
4. Try these commands:

| Command | Tool Used | Expected Duration |
|---------|-----------|-------------------|
| "What time is it?" | `get_current_time` | < 500ms |
| "Flip a coin" | `flip_coin` | < 500ms |
| "Roll a 20 sided die" | `roll_dice` | < 500ms |
| "What's the weather in Tokyo?" | `get_weather` | ~1.5s |
| "Calculate 234 times 567" | `calculator` | ~1s |
| "Search for AI trends" | `web_search` | ~2s |
| "Analyze this text: [long paragraph]" | `analyze_text` | ~4s |
| "Generate a report about AI" | `generate_report` | ~6s |
| "Query the database" | `database_query` | ~5s |

## Project Structure

```
voice_agent_test/
├── voice_agent.py       # Main voice agent with all tools
├── web_server.py        # Web UI server
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variables template
├── .gitignore
└── README.md
```

## API Keys Needed

### Required

- **LiveKit Cloud** (or self-hosted):
  - Sign up at https://cloud.livekit.io
  - Get URL, API Key, and Secret from dashboard

### LLM Provider (choose one)

- **OpenAI**: Get API key from https://platform.openai.com
- **Anthropic**: Get API key from https://console.anthropic.com

### Optional (for specific providers)

- **Deepgram** (STT): https://deepgram.com
- **Cartesia** (TTS): https://cartesia.ai
- **ElevenLabs** (TTS): https://elevenlabs.io

## Success Criteria

The app demonstrates:
- Tools execute WITHOUT the agent announcing them
- Quick tools feel instant (< 500ms)
- Long tools don't break conversation flow
- Complex tool inputs are handled correctly
- Stop button immediately terminates the session
- Agent maintains conversational tone throughout

## Troubleshooting

### "Module not found" errors

```bash
# Make sure you installed dependencies
pip install -r requirements.txt
```

### LiveKit connection errors

- Verify your LIVEKIT_URL, API_KEY, and API_SECRET are correct
- Check that your LiveKit project is active
- Ensure network allows WebSocket connections

### Audio not working

- Check browser microphone permissions
- Verify microphone is not being used by another app
- Try different browser (Chrome recommended)

## Architecture

The agent uses LiveKit Agents framework with:
- **VAD**: Silero for voice activity detection
- **STT**: Deepgram Nova-2 (configurable)
- **LLM**: OpenAI GPT-4o-mini (configurable)
- **TTS**: Cartesia Sonic (configurable)

## License

MIT
