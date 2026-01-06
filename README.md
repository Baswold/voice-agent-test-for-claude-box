# Voice Agent Test Application

A voice agent application to test **clean tool execution** with varying complexity levels. Built with LiveKit Agents (Python).

## Features

- **13 Test Tools** with varying execution times:
  - Quick tools (< 1s): `get_current_time`, `flip_coin`, `roll_dice`
  - Medium tools (1-3s): `get_weather`, `calculator`, `web_search`
  - Long tools (3-10s): `analyze_text`, `generate_report`, `database_query`
  - Additional: `set_agent_mood`, `get_session_stats`, `get_system_info`
  - Edge case: `trigger_error` for testing error handling

- **Silent Tool Execution**: Tools execute WITHOUT the agent announcing them
- **Multiple Interfaces**:
  - **Native Desktop GUI** (Tkinter) - Full-featured with demo options
  - **Headless CLI** - For agent testing and automation
  - **Console Mode** - Interactive LiveKit testing

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/Baswold/voice-agent-test-for-claude-box.git
cd voice-agent-test-for-claude-box

# Create virtual environment
python3 -m venv venv
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

**GUI Mode:**
```bash
python tkinter_ui.py
```

**CLI Mode (Headless - for agent testing):**
```bash
# Text mode with mock responses (fastest for testing)
python cli_tool.py text "What time is it?" "Flip a coin"

# Automated tool testing
python cli_tool.py test --verbose-tool-logging

# LiveKit mode (full audio)
python cli_tool.py livekit --room-name my-room
```

## CLI Usage

The `cli_tool.py` provides a headless CLI with all the same features as the GUI:

### Commands

```bash
# Text mode - headless, no audio, mock responses (fastest for agent testing)
python cli_tool.py text [inputs...]

# Test mode - automated testing of all tools
python cli_tool.py test

# LiveKit mode - full audio with LiveKit
python cli_tool.py livekit

# Console mode - interactive LiveKit console
python cli_tool.py console
```

### Demo Features (Same as GUI)

```bash
# Verbose tool logging
python cli_tool.py text --verbose-tool-logging "Flip a coin"

# Simulated latency (for testing patience)
python cli_tool.py text --simulated-latency --latency-ms 500 "//test flip_coin"

# Announcement mode (for comparison with silent mode)
python cli_tool.py text --announcement-mode "What's the weather?"

# Debug mode
python cli_tool.py text --debug-mode "Get system info"

# Mock mode (fake responses, no API calls)
python cli_tool.py text --mock-mode "Hello"

# JSON output (for agent parsing)
python cli_tool.py text --json-output "What time is it?"

# Combine multiple features
python cli_tool.py text --verbose-tool-logging --simulated-latency --latency-ms 2000 --announcement-mode "//test calculator 234*567"
```

### All CLI Options

| Option | Description |
|--------|-------------|
| `--livekit-url` | LiveKit URL (default: from LIVEKIT_URL env var) |
| `--api-key` | LiveKit API Key |
| `--api-secret` | LiveKit API Secret |
| `--room-name` | Room name (default: voice-agent-test) |
| `--llm` | LLM provider (default: openai/gpt-4o-mini) |
| `--stt` | STT provider (default: local/whisper-base) |
| `--tts` | TTS provider (default: local/macos) |
| `--mood` | Agent mood: friendly, professional, playful, terse |
| `--timeout` | Session timeout in seconds (default: 300) |
| `--output-file` | Write output to file |
| `--json-output` | Output in JSON format |
| `--verbose-tool-logging` | Show detailed tool execution info |
| `--auto-greeting` | Agent greets on session start |
| `--simulated-latency` | Add artificial delay to responses |
| `--latency-ms` | Simulated latency in milliseconds |
| `--sound-effects` | Play sounds on tool execution |
| `--tool-timeline` | Log tool timeline |
| `--transcript-mode` | Show real-time transcription |
| `--debug-mode` | Show internal agent state |
| `--announcement-mode` | Agent announces tools (for comparison) |
| `--no-summary` | Disable conversation summary |
| `--mock-mode` | Use fake responses (faster testing) |
| `--input-file` | Read inputs from file (one per line) |

### Direct Tool Testing

Use `//test` prefix to directly invoke tools:

```bash
python cli_tool.py text "//test get_current_time"
python cli_tool.py text "//test flip_coin"
python cli_tool.py text "//test roll_dice 20"
python cli_tool.py text "//test get_weather Tokyo"
python cli_tool.py text "//test calculator 234*567"
python cli_tool.py text "//test web_search AI trends"
python cli_tool.py text "//test analyze_text I love this product sentiment"
python cli_tool.py text "//test generate_report AI"
python cli_tool.py text "//test database_query"
python cli_tool.py text "//test set_agent_mood playful"
python cli_tool.py text "//test get_session_stats"
python cli_tool.py text "//test trigger_error timeout"
```

## GUI Usage

Launch `python tkinter_ui.py` for the full-featured GUI with:

- Configuration panel for LiveKit and providers
- 11 toggleable demo features
- Tool timeline visualization
- Audio level meter
- Session statistics
- Color-coded conversation log
- Quick test action buttons

## Project Structure

```
voice_agent_test/
├── tkinter_ui.py        # Main GUI application
├── cli_tool.py          # Headless CLI tool
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

### Optional (for cloud providers)

- **Deepgram** (STT): https://deepgram.com
- **Cartesia** (TTS): https://cartesia.ai
- **ElevenLabs** (TTS): https://elevenlabs.io

### Local TTS/STT (included)

- **macOS `say` command** - Built-in on macOS
- **pyttsx3** - Cross-platform TTS
- **Whisper** - Local STT (install with `pip install openai-whisper`)

## Success Criteria

The app demonstrates:
- Tools execute WITHOUT the agent announcing them
- Quick tools feel instant (< 500ms)
- Long tools don't break conversation flow
- Complex tool inputs are handled correctly
- Stop button/CLI termination works immediately
- Agent maintains conversational tone throughout

## Troubleshooting

### "Module not found" errors

```bash
# Make sure you activated the virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

### LiveKit connection errors

- Verify your LIVEKIT_URL, API_KEY, and API_SECRET are correct
- Check that your LiveKit project is active
- Ensure network allows WebSocket connections

### GUI not launching

- Make sure you have Tcl/Tk installed (usually included with Python)
- On Linux: `sudo apt-get install python3-tk`

## License

MIT
