# Voice Agent Test Application

A simple voice agent application to test **clean tool execution** with varying complexity levels. Built with LiveKit Agents (Python) and Tkinter.

## Features

- **10 Test Tools** with varying execution times:
  - Quick tools (< 1s): `get_current_time`, `flip_coin`, `roll_dice`
  - Medium tools (1-3s): `get_weather`, `calculator`, `web_search`
  - Long tools (3-10s): `analyze_text`, `generate_report`, `database_query`
  - Edge case: `trigger_error` for testing error handling

- **Silent Tool Execution**: Tools execute WITHOUT the agent announcing them
- **Native Desktop UI** (Tkinter) with:
  - API key configuration
  - Provider selection (STT/TTS/LLM)
  - Large RED STOP button
  - Real-time status indicator (Listening, Processing, Speaking, Tool Executing)
  - Conversation log with color-coded entries
  - Quick test action buttons

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
# Launch the GUI
python tkinter_ui.py

# Or run in console mode for quick testing
python tkinter_ui.py console
```

## Testing

### GUI Mode

1. Launch `python tkinter_ui.py`
2. Enter your LiveKit credentials in the Configuration panel
3. Click the blue **START** button
4. Try speaking or clicking the quick test buttons:

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

### Console Mode

```bash
python tkinter_ui.py console
```

## Project Structure

```
voice_agent_test/
├── tkinter_ui.py        # Main GUI application with all tools
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

### GUI not launching

- Make sure you have Tcl/Tk installed (usually included with Python)
- On Linux: `sudo apt-get install python3-tk`

## Architecture

The agent uses LiveKit Agents framework with:
- **VAD**: Silero for voice activity detection
- **STT**: Deepgram Nova-2 (configurable)
- **LLM**: OpenAI GPT-4o-mini (configurable)
- **TTS**: Cartesia Sonic (configurable)
- **GUI**: Tkinter with asyncio event loop in background thread

## License

MIT
