#!/usr/bin/env python3
"""
Voice Agent Test CLI - Headless CLI for Agent Testing
A full-featured CLI tool with the same capabilities as the GUI.
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)

# Import local TTS/STT support
from tkinter_ui import (
    get_tts_provider,
    get_stt_provider,
    LocalMacOSTTS,
    LocalPyttsx3TTS,
    PYTTSX3_AVAILABLE,
    WHISPER_AVAILABLE,
)
from livekit.plugins import silero

# Load environment variables
load_dotenv()


# ============================================================================
# CONFIG FROM ENV/CLI
# ============================================================================

class CLIConfig:
    """Configuration from CLI arguments and environment variables."""

    def __init__(self, args=None):
        if args is None:
            args = type('Args', (), {})()

        # Helper to get value with env var fallback
        def get_val(attr_name, env_var=None, default=False, val_type=str):
            val = getattr(args, attr_name, None)
            if val is not None:
                return val
            if env_var:
                env_val = os.getenv(env_var)
                if env_val is not None:
                    if val_type == bool:
                        return env_val.lower() == 'true'
                    elif val_type == int:
                        return int(env_val)
                    return env_val
            return default

        # Demo features - check CLI args first, then env vars, then defaults
        self.verbose_tool_logging = get_val('verbose_tool_logging', 'VERBOSE_TOOL_LOGGING')
        self.auto_greeting = get_val('auto_greeting', 'AUTO_GREETING')
        self.simulated_latency = get_val('simulated_latency', 'SIMULATED_LATENCY')
        self.latency_ms = get_val('latency_ms', 'LATENCY_MS', default=1000, val_type=int)
        self.sound_effects = get_val('sound_effects', 'SOUND_EFFECTS')
        self.tool_timeline = get_val('tool_timeline', 'TOOL_TIMELINE')
        self.transcript_mode = get_val('transcript_mode', 'TRANSCRIPT_MODE')
        self.debug_mode = get_val('debug_mode', 'DEBUG_MODE')
        self.announcement_mode = get_val('announcement_mode', 'ANNOUNCEMENT_MODE')
        # Handle no_summary flag (inverse of conversation_summary)
        no_summary = get_val('no_summary', None)
        if no_summary is not None:
            self.conversation_summary = not no_summary
        else:
            self.conversation_summary = get_val('conversation_summary', 'CONVERSATION_SUMMARY', default=True, val_type=bool)
        self.mock_mode = get_val('mock_mode', 'MOCK_MODE', val_type=bool)

        # Connection settings
        self.livekit_url = get_val('livekit_url', 'LIVEKIT_URL', default='')
        self.api_key = get_val('api_key', 'LIVEKIT_API_KEY', default='')
        self.api_secret = get_val('api_secret', 'LIVEKIT_API_SECRET', default='')
        self.room_name = get_val('room_name', 'ROOM_NAME', default='voice-agent-test')

        # Provider settings
        self.llm_provider = get_val('llm', 'DEFAULT_LLM', default='openai/gpt-4o-mini')
        self.stt_provider = get_val('stt', 'DEFAULT_STT', default='local/whisper-base')
        self.tts_provider = get_val('tts', 'DEFAULT_TTS', default='local/macos')

        # Agent settings
        self.agent_mood = get_val('mood', 'AGENT_MOOD', default='friendly')
        self.timeout = get_val('timeout', 'SESSION_TIMEOUT', default=300, val_type=int)
        self.input_file = getattr(args, 'input_file', None)

        # Output settings
        self.output_file = getattr(args, 'output_file', None)
        self.json_output = getattr(args, 'json_output', False)

        # Session tracking
        self.session_start_time = None
        self.tool_calls = []
        self.user_messages = []
        self.agent_responses = []

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# Global config instance
cli_config = None


# ============================================================================
# OUTPUT LOGGER
# ============================================================================

class CLILogger:
    """Logger for CLI output with optional JSON formatting."""

    def __init__(self, config: CLIConfig):
        self.config = config
        self.events = []

    def log(self, source, message, event_type="info"):
        """Log an event."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        if self.config.json_output:
            event = {
                "timestamp": timestamp,
                "source": source,
                "message": message,
                "event_type": event_type,
            }
            self.events.append(event)
            print(json.dumps(event))
        else:
            colors = {
                "system": "\033[90m",      # Gray
                "user": "\033[94m",        # Blue
                "agent": "\033[92m",       # Green
                "tool": "\033[91m",        # Red
                "error": "\033[95m",       # Magenta
                "debug": "\033[93m",       # Yellow
                "success": "\033[92m",     # Green
            }
            reset = "\033[0m"
            color = colors.get(event_type, "")
            print(f"{color}[{timestamp}] {source}: {message}{reset}")

        # Write to file if specified
        if self.config.output_file:
            with open(self.config.output_file, 'a') as f:
                f.write(f"[{timestamp}] {source}: {message}\n")

    def log_tool_execution(self, tool_name, args, result, duration_ms, is_error=False):
        """Log tool execution."""
        event_type = 'error' if is_error else 'tool'

        if self.config.verbose_tool_logging:
            args_str = str(args) if args else "no args"
            self.log("Tool", f"{tool_name}({args_str}) -> {result} ({duration_ms:.0f}ms)", event_type)
        else:
            self.log("Tool", f"{tool_name} -> {duration_ms:.0f}ms", event_type)

        # Track stats
        self.config.tool_calls.append({
            "tool": tool_name,
            "duration": duration_ms,
            "time": time.time()
        })

    def show_summary(self):
        """Show session summary."""
        if not self.config.conversation_summary:
            return

        duration = int(time.time() - self.config.session_start_time) if self.config.session_start_time else 0
        tool_count = len(self.config.tool_calls)
        avg_latency = sum(t["duration"] for t in self.config.tool_calls) / tool_count if tool_count else 0

        if self.config.json_output:
            summary = {
                "event_type": "summary",
                "duration_seconds": duration,
                "tool_calls": tool_count,
                "user_messages": len(self.config.user_messages),
                "agent_responses": len(self.config.agent_responses),
                "avg_latency_ms": avg_latency,
            }
            print(json.dumps(summary))
        else:
            print("\n" + "="*40)
            print("         SESSION SUMMARY")
            print("="*40)
            print(f"Duration:    {duration}s")
            print(f"Tool Calls:  {tool_count}")
            print(f"User Msgs:   {len(self.config.user_messages)}")
            print(f"Agent Resp:  {len(self.config.agent_responses)}")
            print(f"Avg Latency: {avg_latency:.0f}ms")
            print("="*40 + "\n")


# Global logger instance
cli_logger = None


# ============================================================================
# BASE SYSTEM PROMPTS
# ============================================================================

SILENT_MODE_PROMPT = """
You are a helpful voice assistant. When using tools, NEVER announce that you're using them.

❌ WRONG: "Let me check the weather for you..."
❌ WRONG: "I'll look that up..."
✅ CORRECT: [silently execute tool, then respond with result]

After tool execution, respond naturally with the information.
"""

VERBOSE_MODE_PROMPT = """
You are a helpful voice assistant. When using tools, ALWAYS announce what you're doing.

✅ CORRECT: "Let me check the weather for you..."
✅ CORRECT: "I'll look that up..."
After tool execution, respond naturally with the information.

This is VERBOSE/ANNOUNCEMENT mode for comparison testing.
"""


# ============================================================================
# TEST TOOLS
# ============================================================================

@function_tool
async def get_current_time(context: RunContext[AgentSession], timezone: str = "UTC") -> str:
    """Returns the current time in the user's timezone."""
    start_time = time.time()

    if cli_config.simulated_latency:
        await asyncio.sleep(cli_config.latency_ms / 1000)

    result = datetime.now().astimezone().strftime("%I:%M %p")
    elapsed = (time.time() - start_time) * 1000

    if cli_logger:
        cli_logger.log_tool_execution("get_current_time", timezone, result, elapsed)

    return f"The current time is {result}."


@function_tool
async def flip_coin(context: RunContext[AgentSession]) -> str:
    """Flips a coin and returns heads or tails."""
    start_time = time.time()

    if cli_config.simulated_latency:
        await asyncio.sleep(cli_config.latency_ms / 1000)

    result = random.choice(["heads", "tails"])
    elapsed = (time.time() - start_time) * 1000

    if cli_logger:
        cli_logger.log_tool_execution("flip_coin", None, result, elapsed)

    return f"It's {result}."


@function_tool
async def roll_dice(context: RunContext[AgentSession], sides: int = 6) -> str:
    """Rolls a die with the specified number of sides (default 6)."""
    start_time = time.time()

    if cli_config.simulated_latency:
        await asyncio.sleep(cli_config.latency_ms / 1000)

    if sides < 2 or sides > 100:
        sides = 6
    result = random.randint(1, sides)
    elapsed = (time.time() - start_time) * 1000

    if cli_logger:
        cli_logger.log_tool_execution("roll_dice", f"sides={sides}", result, elapsed)

    return f"You rolled a {result}."


@function_tool
async def get_weather(context: RunContext[AgentSession], location: str) -> str:
    """Gets the current weather for a location."""
    start_time = time.time()
    await asyncio.sleep(1.5)

    if cli_config.simulated_latency:
        await asyncio.sleep(cli_config.latency_ms / 1000)

    conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "windy"]
    temp = random.randint(45, 95)
    condition = random.choice(conditions)
    elapsed = (time.time() - start_time) * 1000

    if cli_logger:
        cli_logger.log_tool_execution("get_weather", location, f"{temp}°F {condition}", elapsed)

    return f"It's currently {temp} degrees Fahrenheit and {condition} in {location}."


@function_tool
async def calculator(context: RunContext[AgentSession], expression: str) -> str:
    """Evaluates a mathematical expression."""
    start_time = time.time()

    if cli_config.simulated_latency:
        await asyncio.sleep(cli_config.latency_ms / 1000)

    try:
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names)
        elapsed = (time.time() - start_time) * 1000

        if cli_logger:
            cli_logger.log_tool_execution("calculator", expression, str(result), elapsed)

        return f"The result is {result}."
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        if cli_logger:
            cli_logger.log_tool_execution("calculator", expression, f"ERROR: {e}", elapsed, is_error=True)
        return f"Sorry, I couldn't calculate that. The error was: {str(e)}"


@function_tool
async def web_search(context: RunContext[AgentSession], query: str) -> str:
    """Searches the web for information."""
    start_time = time.time()
    await asyncio.sleep(2.0)

    if cli_config.simulated_latency:
        await asyncio.sleep(cli_config.latency_ms / 1000)

    results = [
        f"Here's what I found about '{query}':",
        f"1. Wikipedia: {query} is a topic of interest with many applications.",
        f"2. News: Recent updates about {query} have been trending.",
    ]
    elapsed = (time.time() - start_time) * 1000

    if cli_logger:
        cli_logger.log_tool_execution("web_search", query, f"{len(results)} results", elapsed)

    return "\n".join(results)


@function_tool
async def analyze_text(context: RunContext[AgentSession], text: str, analysis_type: str = "sentiment") -> str:
    """Analyzes text using AI. Types: sentiment, summary, keywords, translation."""
    start_time = time.time()
    await asyncio.sleep(4.0)

    if cli_config.simulated_latency:
        await asyncio.sleep(cli_config.latency_ms / 1000)

    text_preview = text[:50] + "..." if len(text) > 50 else text

    if analysis_type == "sentiment":
        sentiments = ["positive", "negative", "neutral"]
        result = f"Sentiment analysis: {random.choice(sentiments)} (confidence: {random.randint(70, 99)}%)"
    elif analysis_type == "summary":
        result = f"Summary: The text discusses key topics and presents various viewpoints."
    elif analysis_type == "keywords":
        keywords = ["technology", "innovation", "analysis", "development", "research", "trends"]
        result = f"Keywords: {', '.join(random.sample(keywords, 3))}"
    else:
        result = f"Analysis complete for '{text_preview}' with type: {analysis_type}"

    elapsed = (time.time() - start_time) * 1000

    if cli_logger:
        cli_logger.log_tool_execution("analyze_text", f"{text_preview} ({analysis_type})", result.split(':')[1] if ':' in result else result, elapsed)

    return result


@function_tool
async def generate_report(context: RunContext[AgentSession], topic: str, sections: list[str], detail_level: str = "brief") -> str:
    """Generates a detailed report on a topic."""
    start_time = time.time()
    await asyncio.sleep(6.0)

    if cli_config.simulated_latency:
        await asyncio.sleep(cli_config.latency_ms / 1000)

    section_names = ", ".join(sections)
    result = f"""Report on {topic}:
Sections: {section_names}
Detail Level: {detail_level}

{' '.join([f"{s.capitalize()}: Analysis and findings for this section." for s in sections[:3]])}

Overall conclusion: The report highlights important aspects of {topic}."""

    elapsed = (time.time() - start_time) * 1000

    if cli_logger:
        cli_logger.log_tool_execution("generate_report", f"{topic} ({len(sections)} sections)", "Complete", elapsed)

    return result


@function_tool
async def database_query(context: RunContext[AgentSession], sql: str, filters: dict[str, Any] | None = None) -> str:
    """Executes a database query with optional filters."""
    start_time = time.time()
    await asyncio.sleep(5.0)

    if cli_config.simulated_latency:
        await asyncio.sleep(cli_config.latency_ms / 1000)

    filters_str = json.dumps(filters) if filters else "none"
    result_count = random.randint(5, 50)
    result = f"Query executed: {sql}\nFilters: {filters_str}\nResults: {result_count} records found."

    elapsed = (time.time() - start_time) * 1000

    if cli_logger:
        cli_logger.log_tool_execution("database_query", sql, f"{result_count} records", elapsed)

    return result


@function_tool
async def trigger_error(context: RunContext[AgentSession], error_type: str = "api_error") -> str:
    """Triggers different types of errors for testing error handling."""
    start_time = time.time()

    error_messages = {
        "timeout": "Request timed out. The service took too long to respond.",
        "api_error": "API error: Invalid credentials or rate limit exceeded.",
        "invalid_input": "Invalid input: The provided parameters are not valid.",
        "network_error": "Network error: Unable to reach the server.",
    }
    elapsed = (time.time() - start_time) * 1000

    if cli_logger:
        cli_logger.log_tool_execution("trigger_error", error_type, f"Simulated: {error_type}", elapsed, is_error=True)

    return f"I encountered an issue: {error_messages.get(error_type, 'Unknown error occurred')}. Please try again."


@function_tool
async def get_system_info(context: RunContext[AgentSession]) -> str:
    """Returns system information for debugging."""
    import platform
    info = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
    }
    return f"System info: {json.dumps(info, indent=2)}"


@function_tool
async def set_agent_mood(context: RunContext[AgentSession], mood: str) -> str:
    """Sets the agent's mood for the conversation. Moods: friendly, professional, playful, terse."""
    valid_moods = ["friendly", "professional", "playful", "terse"]
    if mood.lower() not in valid_moods:
        return f"Please choose a valid mood: {', '.join(valid_moods)}"

    cli_config.agent_mood = mood.lower()
    if cli_logger:
        cli_logger.log("System", f"Agent mood set to: {mood}", "debug")

    return f"I'll now respond in a more {mood} manner."


@function_tool
async def get_session_stats(context: RunContext[AgentSession]) -> str:
    """Returns statistics about the current session."""
    duration = int(time.time() - cli_config.session_start_time) if cli_config.session_start_time else 0
    stats = {
        "duration_seconds": duration,
        "tool_calls": len(cli_config.tool_calls),
        "user_messages": len(cli_config.user_messages),
        "agent_responses": len(cli_config.agent_responses),
        "avg_latency": sum(t["duration"] for t in cli_config.tool_calls) / len(cli_config.tool_calls) if cli_config.tool_calls else 0,
    }
    return f"Session stats: {json.dumps(stats, indent=2)}"


# ============================================================================
# VOICE AGENT
# ============================================================================

class VoiceAgentTestCLI(Agent):
    """A voice agent for CLI testing."""

    def __init__(self, config: CLIConfig):
        # Build instructions based on config
        instructions = ""

        if config.announcement_mode:
            instructions += VERBOSE_MODE_PROMPT
        else:
            instructions += SILENT_MODE_PROMPT

        instructions += f"""

You have access to various tools for different purposes:
- get_current_time: When user asks what time it is
- flip_coin: When user wants to flip a coin
- roll_dice: When user wants to roll dice
- get_weather: When user asks about weather in a location
- calculator: For mathematical calculations
- web_search: When user wants to search for information
- analyze_text: For analyzing text (sentiment, summary, keywords, translation)
- generate_report: For creating structured reports
- database_query: For database queries
- trigger_error: For testing error handling
- set_agent_mood: Change your speaking style (friendly, professional, playful, terse)
- get_session_stats: Get conversation statistics
- get_system_info: Get system information

Current mood: {config.agent_mood}
Be brief and concise in your responses.
"""

        if config.debug_mode:
            instructions += "\n\nDEBUG MODE: You can share internal reasoning when asked."

        super().__init__(instructions=instructions)
        self.mood = config.agent_mood


# ============================================================================
# TEXT-ONLY MODE (for headless testing without audio)
# ============================================================================

class TextOnlyAgent:
    """A text-only agent that processes input and returns responses without audio."""

    def __init__(self, config: CLIConfig, logger: CLILogger):
        self.config = config
        self.logger = logger
        self.llm_provider = config.llm_provider

    async def process_input(self, user_input: str) -> str:
        """Process user input and return the agent's response."""
        from livekit.agents import llm

        self.logger.log("User", user_input, "user")
        self.config.user_messages.append({"text": user_input, "time": time.time()})

        # Check for direct tool commands (for testing)
        if user_input.lower().startswith("//test "):
            test_command = user_input[7:].strip()
            return await self._test_tool(test_command)

        # Get LLM response
        try:
            # For mock mode, return simple responses
            if self.config.mock_mode:
                response = self._mock_response(user_input)
            else:
                # Use LiveKit's LLM for actual responses
                if self.config.announcement_mode:
                    instructions = VERBOSE_MODE_PROMPT
                else:
                    instructions = SILENT_MODE_PROMPT

                # Simple LLM call for text-only mode
                response = await self._get_llm_response(user_input, instructions)

            self.logger.log("Agent", response, "agent")
            self.config.agent_responses.append({"text": response, "time": time.time()})

            return response

        except Exception as e:
            error_msg = f"Error processing input: {str(e)}"
            self.logger.log("Error", error_msg, "error")
            return error_msg

    async def _test_tool(self, test_command: str) -> str:
        """Directly test a tool by name with optional parameters."""
        parts = test_command.split(None, 1)
        tool_name = parts[0]
        params = parts[1] if len(parts) > 1 else ""

        # Tool mapping with parameter handling
        if tool_name == "get_current_time":
            result = await get_current_time(None, params or "UTC")
        elif tool_name == "flip_coin":
            result = await flip_coin(None)
        elif tool_name == "roll_dice":
            sides = int(params) if params.isdigit() else 20
            result = await roll_dice(None, sides)
        elif tool_name == "get_weather":
            result = await get_weather(None, params or "San Francisco")
        elif tool_name == "calculator":
            result = await calculator(None, params or "2+2")
        elif tool_name == "web_search":
            result = await web_search(None, params or "AI trends")
        elif tool_name == "analyze_text":
            parts2 = params.split(None, 1) if params else []
            text = parts2[0] if len(parts2) > 0 else "I love this product"
            analysis_type = parts2[1] if len(parts2) > 1 else "sentiment"
            result = await analyze_text(None, text, analysis_type)
        elif tool_name == "generate_report":
            result = await generate_report(None, params or "AI", ["intro", "analysis"], "brief")
        elif tool_name == "database_query":
            result = await database_query(None, params or "SELECT * FROM users", {"status": "active"})
        elif tool_name == "trigger_error":
            result = await trigger_error(None, params or "api_error")
        elif tool_name == "get_system_info":
            result = await get_system_info(None)
        elif tool_name == "get_session_stats":
            result = await get_session_stats(None)
        elif tool_name == "set_agent_mood":
            result = await set_agent_mood(None, params or "friendly")
        else:
            available = "get_current_time, flip_coin, roll_dice, get_weather, calculator, web_search, analyze_text, generate_report, database_query, trigger_error, get_system_info, get_session_stats, set_agent_mood"
            return f"[TOOL TEST] Unknown tool: {tool_name}. Available: {available}"

        return f"[TOOL TEST] {tool_name}: {result}"

    def _mock_response(self, user_input: str) -> str:
        """Generate mock responses for testing."""
        responses = {
            "time": "It's 3:45 PM.",
            "hello": "Hello! How can I help you today?",
            "hi": "Hi there! What can I do for you?",
            "weather": "It's currently sunny and 72°F here.",
            "coin": "It's heads!",
            "dice": "You rolled a 16.",
            "bye": "Goodbye! Have a great day.",
        }

        user_input_lower = user_input.lower()
        for key, response in responses.items():
            if key in user_input_lower:
                return response

        return "I understand. How else can I help you?"

    async def _get_llm_response(self, user_input: str, instructions: str) -> str:
        """Get response from LLM."""
        # Import OpenAI for text-only mode
        try:
            from openai import AsyncOpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return self._mock_response(user_input)

            client = AsyncOpenAI(api_key=api_key)

            # Determine model from provider string
            provider = self.config.llm_provider.split("/")[0]
            model_map = {
                "openai": "gpt-4o-mini",
                "anthropic": "gpt-4o-mini",  # Fallback to OpenAI if Anthropic
            }
            model = model_map.get(provider, "gpt-4o-mini")

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": instructions.strip()},
                    {"role": "user", "content": user_input},
                ],
                max_tokens=200,
            )

            return response.choices[0].message.content

        except Exception as e:
            if self.config.debug_mode:
                return f"LLM Error: {str(e)}"
            return self._mock_response(user_input)


# ============================================================================
# CLI ENTRY POINTS
# ============================================================================

async def cli_entrypoint(ctx: JobContext) -> None:
    """Entrypoint for CLI mode with LiveKit."""
    global cli_config, cli_logger

    cli_config.session_start_time = time.time()
    cli_logger = CLILogger(cli_config)

    cli_logger.log("System", f"Starting voice agent session in room: {cli_config.room_name}", "system")
    cli_logger.log("Config", f"LLM: {cli_config.llm_provider}, STT: {cli_config.stt_provider}, TTS: {cli_config.tts_provider}", "debug")
    cli_logger.log("Config", f"Mood: {cli_config.agent_mood}, Timeout: {cli_config.timeout}s", "debug")

    # List enabled features
    enabled_features = [k for k, v in cli_config.to_dict().items() if v is True and not k.startswith('_')]
    if enabled_features:
        cli_logger.log("Config", f"Enabled features: {', '.join(enabled_features)}", "debug")

    await ctx.connect()

    # Get providers
    stt_instance = get_stt_provider(cli_config.stt_provider)
    tts_instance = get_tts_provider(cli_config.tts_provider)

    cli_logger.log("System", f"STT: {type(stt_instance).__name__}", "system")
    cli_logger.log("System", f"TTS: {type(tts_instance).__name__}", "system")

    # Create agent
    agent = VoiceAgentTestCLI(cli_config)

    # Create session
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=stt_instance,
        llm=cli_config.llm_provider,
        tts=tts_instance,
    )

    # Auto greeting
    if cli_config.auto_greeting:
        await session.say("Hello! How can I help you test my tool capabilities today?", allow_interruptions=True)

    cli_logger.log("System", "Connected. Listening...", "success")

    # Run with timeout
    try:
        await session.start(agent=agent, room=ctx.room)

        # Wait for timeout or manual stop
        start_time = time.time()
        while time.time() - start_time < cli_config.timeout:
            await asyncio.sleep(1)

    except Exception as e:
        cli_logger.log("Error", str(e), "error")
    finally:
        cli_logger.show_summary()
        cli_logger.log("System", "Session ended.", "system")


async def text_mode(args):
    """Run in text-only mode (headless, no audio)."""
    global cli_config, cli_logger

    cli_config = CLIConfig(args)
    cli_logger = CLILogger(cli_config)

    cli_logger.log("System", "Voice Agent CLI - Text Mode", "system")
    cli_logger.log("Config", f"LLM: {cli_config.llm_provider}, Mood: {cli_config.agent_mood}", "debug")

    # List enabled features
    enabled_features = [k for k, v in cli_config.to_dict().items() if v is True and not k.startswith('_')]
    if enabled_features:
        cli_logger.log("Config", f"Enabled features: {', '.join(enabled_features)}", "debug")

    agent = TextOnlyAgent(cli_config, cli_logger)

    # Process input from file or stdin
    if cli_config.input_file:
        with open(cli_config.input_file, 'r') as f:
            inputs = [line.strip() for line in f if line.strip()]
    else:
        # Read from stdin until EOF
        if args.inputs:
            inputs = args.inputs
        else:
            cli_logger.log("System", "Enter inputs (one per line), Ctrl-D to finish:", "system")
            inputs = []
            try:
                for line in sys.stdin:
                    inputs.append(line.strip())
            except KeyboardInterrupt:
                pass

    # Auto greeting
    if cli_config.auto_greeting:
        greeting = await agent.process_input("Hello")
        print(f"\nAgent: {greeting}\n")

    # Process each input
    for user_input in inputs:
        if not user_input:
            continue

        response = await agent.process_input(user_input)

        if cli_config.json_output:
            print(json.dumps({"user_input": user_input, "agent_response": response}))
        else:
            print(f"Agent: {response}\n")

    # Show summary
    cli_logger.show_summary()


async def test_mode(args):
    """Run automated tests on all tools."""
    global cli_config, cli_logger

    cli_config = CLIConfig(args)
    cli_logger = CLILogger(cli_config)

    cli_logger.log("System", "Voice Agent CLI - Test Mode", "system")
    cli_logger.log("System", "Running automated tests on all tools...\n", "system")

    agent = TextOnlyAgent(cli_config, cli_logger)

    # Test all tools
    tests = [
        ("get_current_time", ""),
        ("flip_coin", ""),
        ("roll_dice", ""),
        ("get_weather", "San Francisco"),
        ("calculator", "234 * 567"),
        ("web_search", "AI trends"),
        ("analyze_text", "I love this product"),
        ("generate_report", ""),
        ("database_query", ""),
    ]

    for tool, param in tests:
        if param:
            test_input = f"//test {tool} {param}"
        else:
            test_input = f"//test {tool}"

        response = await agent.process_input(test_input)
        print(f"{response}\n")

    cli_logger.show_summary()


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def create_parser():
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Voice Agent Test CLI - Headless CLI for Agent Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in text mode with prompts
  python cli_tool.py text --inputs "What time is it?" "Flip a coin"

  # Run in text mode with input file
  python cli_tool.py text --input-file prompts.txt

  # Run automated tests
  python cli_tool.py test --verbose-tool-logging

  # Run with LiveKit (full audio)
  python cli_tool.py livekit --room-name test-room

  # Enable demo features
  python cli_tool.py text --simulated-latency --latency-ms 500 --announcement-mode
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Common arguments
    def add_common_args(subparser):
        """Add common arguments to a subparser."""
        subparser.add_argument('--livekit-url', default=None, help='LiveKit URL (default: from LIVEKIT_URL env var)')
        subparser.add_argument('--api-key', default=None, help='LiveKit API Key (default: from LIVEKIT_API_KEY env var)')
        subparser.add_argument('--api-secret', default=None, help='LiveKit API Secret (default: from LIVEKIT_API_SECRET env var)')
        subparser.add_argument('--room-name', default=None, help='Room name (default: from ROOM_NAME env var)')
        subparser.add_argument('--llm', default=None, help='LLM provider (default: from DEFAULT_LLM env var)')
        subparser.add_argument('--stt', default=None, help='STT provider (default: from DEFAULT_STT env var)')
        subparser.add_argument('--tts', default=None, help='TTS provider (default: from DEFAULT_TTS env var)')
        subparser.add_argument('--mood', choices=['friendly', 'professional', 'playful', 'terse'], default=None, help='Agent mood (default: from AGENT_MOOD env var)')
        subparser.add_argument('--timeout', type=int, default=None, help='Session timeout in seconds (default: from SESSION_TIMEOUT env var)')
        subparser.add_argument('--output-file', help='Write output to file')
        subparser.add_argument('--json-output', action='store_true', help='Output in JSON format')

        # Demo features (all default to None to check env vars)
        subparser.add_argument('--verbose-tool-logging', action='store_true', default=None, help='Show detailed tool execution info')
        subparser.add_argument('--auto-greeting', action='store_true', default=None, help='Agent greets on session start')
        subparser.add_argument('--simulated-latency', action='store_true', default=None, help='Add artificial delay to responses')
        subparser.add_argument('--latency-ms', type=int, default=None, help='Simulated latency in milliseconds (default: from LATENCY_MS env var)')
        subparser.add_argument('--sound-effects', action='store_true', default=None, help='Play sounds on tool execution')
        subparser.add_argument('--tool-timeline', action='store_true', default=None, help='Log tool timeline')
        subparser.add_argument('--transcript-mode', action='store_true', default=None, help='Show real-time transcription')
        subparser.add_argument('--debug-mode', action='store_true', default=None, help='Show internal agent state')
        subparser.add_argument('--announcement-mode', action='store_true', default=None, help='Agent announces tools (for comparison)')
        subparser.add_argument('--no-summary', action='store_true', default=None, dest='no_summary', help='Disable conversation summary')
        subparser.add_argument('--mock-mode', action='store_true', default=None, help='Use fake responses (faster testing)')

    # Text mode command
    text_parser = subparsers.add_parser('text', help='Run in text-only mode (headless, no audio)')
    add_common_args(text_parser)
    text_parser.add_argument('--input-file', help='Read inputs from file (one per line)')
    text_parser.add_argument('inputs', nargs='*', help='Input prompts to process (use quotes for multi-word prompts)')

    # Test mode command
    test_parser = subparsers.add_parser('test', help='Run automated tests on all tools')
    add_common_args(test_parser)

    # LiveKit mode command
    livekit_parser = subparsers.add_parser('livekit', help='Run with LiveKit (full audio)')
    add_common_args(livekit_parser)

    # Console mode command (for interactive testing)
    console_parser = subparsers.add_parser('console', help='Run in interactive console mode')
    add_common_args(console_parser)

    return parser


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    parser = create_parser()

    # If no args, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    global cli_config, cli_logger

    # Initialize config
    cli_config = CLIConfig(args)
    cli_logger = CLILogger(cli_config)

    if args.command == 'text':
        # Run text mode (async)
        asyncio.run(text_mode(args))

    elif args.command == 'test':
        # Run test mode (async)
        asyncio.run(test_mode(args))

    elif args.command == 'livekit':
        # Run with LiveKit
        from livekit.agents import cli
        cli.run_app(WorkerOptions(entrypoint_fnc=cli_entrypoint))

    elif args.command == 'console':
        # Run console mode
        from livekit.agents import cli
        cli.run_app(WorkerOptions(entrypoint_fnc=cli_entrypoint))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
