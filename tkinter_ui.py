#!/usr/bin/env python3
"""
Voice Agent Test Application - Tkinter GUI
A simple voice agent to test clean tool execution with varying complexity levels.
"""

import asyncio
import json
import os
import random
import threading
import time
from datetime import datetime
from tkinter import ttk, filedialog, messagebox
from typing import Any

import tkinter as tk
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
from livekit.plugins import deepgram, openai, cartesia, silero

# Load environment variables
load_dotenv()

# Critical system prompt for silent tool execution
SILENT_TOOL_INSTRUCTION = """
You are a helpful voice assistant. When using tools, NEVER announce that you're using them.

❌ WRONG: "Let me check the weather for you..."
❌ WRONG: "I'll look that up..."
❌ WRONG: "Calling the weather function..."
✅ CORRECT: [silently execute tool, then respond with result]

After tool execution, respond naturally with the information:
"It's currently 72°F and sunny in San Francisco."

Never mention that you're calling a function or using a tool. Just execute the tool and give the user the result directly.
"""


# ============================================================================
# TEST TOOLS
# ============================================================================

@function_tool
async def get_current_time(context: RunContext[AgentSession], timezone: str = "UTC") -> str:
    """Returns the current time in the user's timezone."""
    start_time = time.time()
    result = datetime.now().astimezone().strftime("%I:%M %p")
    elapsed = (time.time() - start_time) * 1000
    if gui_instance:
        gui_instance.log_tool(f"get_current_time({timezone}) -> {result} ({elapsed:.0f}ms)")
    return f"The current time is {result}."


@function_tool
async def flip_coin(context: RunContext[AgentSession]) -> str:
    """Flips a coin and returns heads or tails."""
    start_time = time.time()
    result = random.choice(["heads", "tails"])
    elapsed = (time.time() - start_time) * 1000
    if gui_instance:
        gui_instance.log_tool(f"flip_coin() -> {result} ({elapsed:.0f}ms)")
    return f"It's {result}."


@function_tool
async def roll_dice(context: RunContext[AgentSession], sides: int = 6) -> str:
    """Rolls a die with the specified number of sides (default 6)."""
    start_time = time.time()
    if sides < 2 or sides > 100:
        sides = 6
    result = random.randint(1, sides)
    elapsed = (time.time() - start_time) * 1000
    if gui_instance:
        gui_instance.log_tool(f"roll_dice({sides}) -> {result} ({elapsed:.0f}ms)")
    return f"You rolled a {result}."


@function_tool
async def get_weather(context: RunContext[AgentSession], location: str) -> str:
    """Gets the current weather for a location."""
    start_time = time.time()
    await asyncio.sleep(1.5)
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "windy"]
    temp = random.randint(45, 95)
    condition = random.choice(conditions)
    elapsed = (time.time() - start_time) * 1000
    if gui_instance:
        gui_instance.log_tool(f"get_weather('{location}') -> {temp}°F {condition} ({elapsed:.0f}ms)")
    return f"It's currently {temp} degrees Fahrenheit and {condition} in {location}."


@function_tool
async def calculator(context: RunContext[AgentSession], expression: str) -> str:
    """Evaluates a mathematical expression."""
    start_time = time.time()
    try:
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names)
        elapsed = (time.time() - start_time) * 1000
        if gui_instance:
            gui_instance.log_tool(f"calculator('{expression}') -> {result} ({elapsed:.0f}ms)")
        return f"The result is {result}."
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        if gui_instance:
            gui_instance.log_tool(f"calculator('{expression}') -> ERROR: {e} ({elapsed:.0f}ms)")
        return f"Sorry, I couldn't calculate that. The error was: {str(e)}"


@function_tool
async def web_search(context: RunContext[AgentSession], query: str) -> str:
    """Searches the web for information."""
    start_time = time.time()
    await asyncio.sleep(2.0)
    results = [
        f"Here's what I found about '{query}':",
        f"1. Wikipedia: {query} is a topic of interest with many applications.",
        f"2. News: Recent updates about {query} have been trending.",
    ]
    elapsed = (time.time() - start_time) * 1000
    if gui_instance:
        gui_instance.log_tool(f"web_search('{query}') -> {len(results)} results ({elapsed:.0f}ms)")
    return "\n".join(results)


@function_tool
async def analyze_text(context: RunContext[AgentSession], text: str, analysis_type: str = "sentiment") -> str:
    """Analyzes text using AI. Types: sentiment, summary, keywords, translation."""
    start_time = time.time()
    await asyncio.sleep(4.0)
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
    if gui_instance:
        gui_instance.log_tool(f"analyze_text('{text_preview}', '{analysis_type}') ({elapsed:.0f}ms)")
    return result


@function_tool
async def generate_report(context: RunContext[AgentSession], topic: str, sections: list[str], detail_level: str = "brief") -> str:
    """Generates a detailed report on a topic."""
    start_time = time.time()
    await asyncio.sleep(6.0)
    section_names = ", ".join(sections)
    result = f"""Report on {topic}:
Sections: {section_names}
Detail Level: {detail_level}

{' '.join([f"{s.capitalize()}: Analysis and findings for this section." for s in sections[:3]])}

Overall conclusion: The report highlights important aspects of {topic}."""

    elapsed = (time.time() - start_time) * 1000
    if gui_instance:
        gui_instance.log_tool(f"generate_report('{topic}', {sections}, '{detail_level}') ({elapsed:.0f}ms)")
    return result


@function_tool
async def database_query(context: RunContext[AgentSession], sql: str, filters: dict[str, Any] | None = None) -> str:
    """Executes a database query with optional filters."""
    start_time = time.time()
    await asyncio.sleep(5.0)
    filters_str = json.dumps(filters) if filters else "none"
    result_count = random.randint(5, 50)
    result = f"Query executed: {sql}\nFilters: {filters_str}\nResults: {result_count} records found."

    elapsed = (time.time() - start_time) * 1000
    if gui_instance:
        gui_instance.log_tool(f"database_query('{sql}', {filters_str}) -> {result_count} records ({elapsed:.0f}ms)")
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
    if gui_instance:
        gui_instance.log_tool(f"trigger_error('{error_type}') -> simulating error ({elapsed:.0f}ms)")
    return f"I encountered an issue: {error_messages.get(error_type, 'Unknown error occurred')}. Please try again."


# ============================================================================
# VOICE AGENT
# ============================================================================

class VoiceAgentTest(Agent):
    """A voice agent for testing clean tool execution."""

    def __init__(self, llm_provider: str = "openai/gpt-4o-mini"):
        instructions = SILENT_TOOL_INSTRUCTION + """

You have access to various tools for different purposes. Use them appropriately:
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

Be friendly, concise, and natural in your responses. Remember: NEVER announce you're using a tool!"""
        super().__init__(instructions=instructions)


# Global GUI instance for tool logging
gui_instance = None


# ============================================================================
# ASYNCIO RUNNER FOR TKINTER
# ============================================================================

class AsyncioRunner:
    """Runs asyncio event loop in a separate thread."""

    def __init__(self):
        self.loop = None
        self.thread = None
        self.running = False

    def start(self):
        """Start the asyncio event loop in a background thread."""
        self.loop = asyncio.new_event_loop()
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        """Run the event loop."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def stop(self):
        """Stop the event loop."""
        self.running = False
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=2)

    def run_coroutine(self, coro):
        """Run a coroutine from the main thread."""
        if self.loop:
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            return future
        return None


# Global asyncio runner
async_runner = AsyncioRunner()


# ============================================================================
# TKINTER GUI
# ============================================================================

class VoiceAgentGUI:
    """Main GUI application for Voice Agent Test."""

    def __init__(self, root):
        global gui_instance
        gui_instance = self

        self.root = root
        self.root.title("Voice Agent Test - Clean Tool Execution")
        self.root.geometry("900x700")
        self.root.configure(bg="#1a1a2e")

        # State
        self.is_running = False
        self.session = None

        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Configure colors
        self.colors = {
            'bg': '#1a1a2e',
            'card_bg': '#2a2a4e',
            'text': '#ffffff',
            'accent': '#4a9eff',
            'success': '#00ff88',
            'warning': '#ffa500',
            'danger': '#ff4757',
            'muted': '#888888',
        }

        # Build UI
        self._build_header()
        self._build_config_panel()
        self._build_status_panel()
        self._build_log_panel()
        self._build_quick_actions()

        # Load saved config
        self._load_config()

        # Start asyncio runner
        async_runner.start()

    def _build_header(self):
        """Build the header section."""
        header = tk.Frame(self.root, bg="#1a1a2e", pady=10)
        header.pack(fill='x')

        title = tk.Label(
            header,
            text="🎙️ Voice Agent Test",
            font=("Arial", 24, "bold"),
            bg="#1a1a2e",
            fg="#ffffff"
        )
        title.pack()

        subtitle = tk.Label(
            header,
            text="Clean Tool Execution Demonstration",
            font=("Arial", 12),
            bg="#1a1a2e",
            fg="#888888"
        )
        subtitle.pack()

    def _build_config_panel(self):
        """Build the configuration panel."""
        config_frame = tk.Frame(self.root, bg=self.colors['card_bg'], padx=20, pady=15)
        config_frame.pack(fill='x', padx=10, pady=10)

        # Title
        tk.Label(
            config_frame,
            text="🔧 Configuration",
            font=("Arial", 12, "bold"),
            bg=self.colors['card_bg'],
            fg=self.colors['accent']
        ).pack(anchor='w')

        # LiveKit settings
        lk_frame = tk.Frame(config_frame, bg=self.colors['card_bg'])
        lk_frame.pack(fill='x', pady=5)

        tk.Label(lk_frame, text="LiveKit URL:", bg=self.colors['card_bg'], fg=self.colors['text'], width=15, anchor='w').grid(row=0, column=0, sticky='w')
        self.livekit_url = tk.Entry(lk_frame, bg="#3a3a5e", fg="white", insertbackground='white')
        self.livekit_url.grid(row=0, column=1, sticky='ew', padx=5)

        tk.Label(lk_frame, text="API Key:", bg=self.colors['card_bg'], fg=self.colors['text'], width=15, anchor='w').grid(row=0, column=2, sticky='w')
        self.api_key = tk.Entry(lk_frame, bg="#3a3a5e", fg="white", insertbackground='white', show='*')
        self.api_key.grid(row=0, column=3, sticky='ew', padx=5)

        tk.Label(lk_frame, text="API Secret:", bg=self.colors['card_bg'], fg=self.colors['text'], width=15, anchor='w').grid(row=1, column=0, sticky='w')
        self.api_secret = tk.Entry(lk_frame, bg="#3a3a5e", fg="white", insertbackground='white', show='*')
        self.api_secret.grid(row=1, column=1, sticky='ew', padx=5)

        tk.Label(lk_frame, text="Room Name:", bg=self.colors['card_bg'], fg=self.colors['text'], width=15, anchor='w').grid(row=1, column=2, sticky='w')
        self.room_name = tk.Entry(lk_frame, bg="#3a3a5e", fg="white", insertbackground='white')
        self.room_name.grid(row=1, column=3, sticky='ew', padx=5)

        lk_frame.columnconfigure(1, weight=1)
        lk_frame.columnconfigure(3, weight=1)

        # Provider settings
        provider_frame = tk.Frame(config_frame, bg=self.colors['card_bg'])
        provider_frame.pack(fill='x', pady=5)

        tk.Label(provider_frame, text="LLM Provider:", bg=self.colors['card_bg'], fg=self.colors['text'], width=15, anchor='w').grid(row=0, column=0, sticky='w')
        self.llm_provider = ttk.Combobox(provider_frame, values=[
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "openai/o3-mini",
            "anthropic/claude-3-haiku",
            "anthropic/claude-3.5-sonnet",
        ], state='readonly')
        self.llm_provider.set("openai/gpt-4o-mini")
        self.llm_provider.grid(row=0, column=1, sticky='ew', padx=5)

        tk.Label(provider_frame, text="STT Provider:", bg=self.colors['card_bg'], fg=self.colors['text'], width=15, anchor='w').grid(row=0, column=2, sticky='w')
        self.stt_provider = ttk.Combobox(provider_frame, values=[
            "deepgram/nova-2",
            "deepgram/nova-3",
        ], state='readonly')
        self.stt_provider.set("deepgram/nova-2")
        self.stt_provider.grid(row=0, column=3, sticky='ew', padx=5)

        tk.Label(provider_frame, text="TTS Provider:", bg=self.colors['card_bg'], fg=self.colors['text'], width=15, anchor='w').grid(row=1, column=0, sticky='w')
        self.tts_provider = ttk.Combobox(provider_frame, values=[
            "cartesia/sonic-english",
            "cartesia/sonic-2",
            "elevenlabs/turbo-2",
        ], state='readonly')
        self.tts_provider.set("cartesia/sonic-english")
        self.tts_provider.grid(row=1, column=1, sticky='ew', padx=5)

        provider_frame.columnconfigure(1, weight=1)
        provider_frame.columnconfigure(3, weight=1)

    def _build_status_panel(self):
        """Build the status and control panel."""
        status_frame = tk.Frame(self.root, bg=self.colors['card_bg'], padx=20, pady=15)
        status_frame.pack(fill='x', padx=10, pady=10)

        tk.Label(
            status_frame,
            text="📊 Session Status",
            font=("Arial", 12, "bold"),
            bg=self.colors['card_bg'],
            fg=self.colors['accent']
        ).pack(anchor='w')

        # Status indicator
        self.status_container = tk.Frame(status_frame, bg=self.colors['card_bg'])
        self.status_container.pack(pady=10)

        self.status_dot = tk.Canvas(
            self.status_container,
            width=20,
            height=20,
            bg=self.colors['card_bg'],
            highlightthickness=0
        )
        self.status_dot.grid(row=0, column=0, padx=10)
        self._draw_status_dot('#555555')

        self.status_text = tk.Label(
            self.status_container,
            text="Disconnected",
            font=("Arial", 14),
            bg=self.colors['card_bg'],
            fg="#ffffff"
        )
        self.status_text.grid(row=0, column=1)

        # Control buttons
        button_frame = tk.Frame(status_frame, bg=self.colors['card_bg'])
        button_frame.pack(pady=10)

        self.start_btn = tk.Button(
            button_frame,
            text="🎤 START",
            font=("Arial", 14, "bold"),
            bg=self.colors['accent'],
            fg="white",
            width=15,
            height=2,
            cursor="hand2",
            command=self.start_session
        )
        self.start_btn.pack(side='left', padx=10)

        self.stop_btn = tk.Button(
            button_frame,
            text="⏹ STOP",
            font=("Arial", 16, "bold"),
            bg=self.colors['danger'],
            fg="white",
            width=15,
            height=2,
            cursor="hand2",
            command=self.stop_session,
            state='disabled'
        )
        self.stop_btn.pack(side='left', padx=10)

    def _build_log_panel(self):
        """Build the conversation log panel."""
        log_frame = tk.Frame(self.root, bg=self.colors['card_bg'], padx=20, pady=15)
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)

        tk.Label(
            log_frame,
            text="💬 Conversation Log",
            font=("Arial", 12, "bold"),
            bg=self.colors['card_bg'],
            fg=self.colors['accent']
        ).pack(anchor='w')

        # Log text widget with scrollbar
        log_container = tk.Frame(log_frame, bg=self.colors['card_bg'])
        log_container.pack(fill='both', expand=True, pady=10)

        scrollbar = tk.Scrollbar(log_container, bg="#3a3a5e")
        scrollbar.pack(side='right', fill='y')

        self.log_text = tk.Text(
            log_container,
            bg="#0a0a1e",
            fg="#00ff88",
            font=("Consolas", 10),
            yscrollcommand=scrollbar.set,
            insertbackground='white',
            relief='flat',
            padx=10,
            pady=10
        )
        self.log_text.pack(fill='both', expand=True)
        scrollbar.config(command=self.log_text.yview)

        # Configure tags for different log types
        self.log_text.tag_config('system', foreground='#888888')
        self.log_text.tag_config('user', foreground='#4a9eff')
        self.log_text.tag_config('agent', foreground='#00ff88')
        self.log_text.tag_config('tool', foreground='#ff6b6b')
        self.log_text.tag_config('error', foreground='#ff4757')

        self.log("System", "Ready to start. Configure your settings above and click START.", "system")

        # Clear button
        clear_btn = tk.Button(
            log_frame,
            text="Clear Log",
            font=("Arial", 10),
            bg="#3a3a5e",
            fg="white",
            cursor="hand2",
            command=self.clear_log
        )
        clear_btn.pack(anchor='e')

    def _build_quick_actions(self):
        """Build quick action buttons for testing."""
        actions_frame = tk.Frame(self.root, bg=self.colors['card_bg'], padx=20, pady=15)
        actions_frame.pack(fill='x', padx=10, pady=10)

        tk.Label(
            actions_frame,
            text="⚡ Quick Test Actions (click to speak)",
            font=("Arial", 12, "bold"),
            bg=self.colors['card_bg'],
            fg=self.colors['accent']
        ).pack(anchor='w')

        buttons_frame = tk.Frame(actions_frame, bg=self.colors['card_bg'])
        buttons_frame.pack(fill='x', pady=10)

        actions = [
            ("What time is it?", "get_current_time"),
            ("Flip a coin", "flip_coin"),
            ("Roll d20", "roll_dice"),
            ("Weather in Tokyo", "get_weather"),
            ("234 * 567", "calculator"),
            ("Search AI trends", "web_search"),
            ("Sentiment: I love this", "analyze_text"),
        ]

        for i, (label, tool) in enumerate(actions):
            btn = tk.Button(
                buttons_frame,
                text=label,
                font=("Arial", 10),
                bg="#3a3a5e",
                fg="white",
                cursor="hand2",
                command=lambda l=label: self.speak_test(l)
            )
            btn.grid(row=i//4, column=i%4, padx=5, pady=5, sticky='ew')

        buttons_frame.columnconfigure(0, weight=1)
        buttons_frame.columnconfigure(1, weight=1)
        buttons_frame.columnconfigure(2, weight=1)
        buttons_frame.columnconfigure(3, weight=1)

    def _draw_status_dot(self, color):
        """Draw the status indicator dot."""
        self.status_dot.delete("all")
        self.status_dot.create_oval(2, 2, 18, 18, fill=color, outline="")

    def _load_config(self):
        """Load saved configuration."""
        import json
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
                self.livekit_url.insert(0, config.get('livekit_url', ''))
                self.api_key.insert(0, config.get('api_key', ''))
                self.api_secret.insert(0, config.get('api_secret', ''))
                self.room_name.insert(0, config.get('room_name', 'voice-agent-test'))
                self.llm_provider.set(config.get('llm_provider', 'openai/gpt-4o-mini'))
                self.stt_provider.set(config.get('stt_provider', 'deepgram/nova-2'))
                self.tts_provider.set(config.get('tts_provider', 'cartesia/sonic-english'))
        except FileNotFoundError:
            self.room_name.insert(0, 'voice-agent-test')

    def _save_config(self):
        """Save current configuration."""
        import json
        config = {
            'livekit_url': self.livekit_url.get(),
            'api_key': self.api_key.get(),
            'api_secret': self.api_secret.get(),
            'room_name': self.room_name.get(),
            'llm_provider': self.llm_provider.get(),
            'stt_provider': self.stt_provider.get(),
            'tts_provider': self.tts_provider.get(),
        }
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def log(self, source, message, tag='system'):
        """Add a log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert('end', f"[{timestamp}] ", 'system')
        self.log_text.insert('end', f"{source}: ", tag)
        self.log_text.insert('end', f"{message}\n")
        self.log_text.see('end')

    def log_tool(self, message):
        """Log a tool execution from the agent."""
        self.log("Tool", message, "tool")

    def log_user(self, message):
        """Log a user message."""
        self.log("User", message, "user")

    def log_agent(self, message):
        """Log an agent message."""
        self.log("Agent", message, "agent")

    def clear_log(self):
        """Clear the log."""
        self.log_text.delete(1.0, 'end')

    def set_status(self, status, text, color):
        """Update the status indicator."""
        self.root.after(0, lambda: self._draw_status_dot(color))
        self.root.after(0, lambda: self.status_text.config(text=text))

    def start_session(self):
        """Start the voice session."""
        self._save_config()

        livekit_url = self.livekit_url.get()
        api_key = self.api_key.get()
        api_secret = self.api_secret.get()

        if not all([livekit_url, api_key, api_secret]):
            messagebox.showerror("Error", "Please fill in LiveKit URL, API Key, and API Secret")
            return

        self.is_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.set_status('connecting', 'Connecting...', '#ffa500')

        self.log("System", "Starting voice session...", "system")

        # Run the agent in the asyncio thread
        async def run_agent():
            try:
                from livekit import rtc

                # Create room
                room = rtc.Room()
                await room.connect(
                    livekit_url,
                    self._create_token(api_key, api_secret, self.room_name.get())
                )

                self.set_status('listening', 'Listening', '#00ff88')
                self.log("System", "Connected to room. Listening...", "system")

                # Keep running until stopped
                while self.is_running:
                    await asyncio.sleep(0.1)

                await room.disconnect()

            except Exception as e:
                self.log("System", f"Error: {str(e)}", "error")
                self.set_status('disconnected', 'Connection Failed', '#ff4757')
                self.start_btn.config(state='normal')
                self.stop_btn.config(state='disabled')

        async_runner.run_coroutine(run_agent())

    def _create_token(self, api_key, api_secret, room_name):
        """Create a JWT token for LiveKit."""
        from livekit import api
        import time

        token = api.AccessToken(api_key, api_secret) \
            .with_identity("user") \
            .with_name("User") \
            .with_grants(api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )) \
            .with_validity(int(time.time()) + 3600)

        return token.to_jwt()

    def stop_session(self):
        """Stop the voice session."""
        self.is_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.set_status('disconnected', 'Disconnected', '#555555')
        self.log("System", "Session ended.", "system")

    def speak_test(self, text):
        """Send a test phrase."""
        if self.is_running:
            self.log_user(text)
            # This would send text to the agent via data channel
            # For console mode, just log it
        else:
            self.log("System", "Please start the session first", "system")


# ============================================================================
# CONSOLE MODE (for testing without GUI)
# ============================================================================

async def console_entrypoint(ctx: JobContext) -> None:
    """Entrypoint for console mode testing."""
    await ctx.connect()

    global gui_instance

    llm_provider = os.getenv("DEFAULT_LLM", "openai/gpt-4o-mini")
    stt_provider = os.getenv("DEFAULT_STT", "deepgram/nova-2")
    tts_provider = os.getenv("DEFAULT_TTS", "cartesia/sonic-english")

    print(f"[CONFIG] LLM: {llm_provider}, STT: {stt_provider}, TTS: {tts_provider}")

    agent = VoiceAgentTest(llm_provider=llm_provider)

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=stt_provider,
        llm=llm_provider,
        tts=tts_provider,
    )

    print("[SESSION] Starting agent session...")

    await session.start(agent=agent, room=ctx.room)
    print("[SESSION] Ready to listen. Waiting for user input...")


class ConsoleLogger:
    """Simple console logger for tool calls."""
    def log_tool(self, message):
        print(f"[TOOL] {message}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "console":
        # Console mode for testing
        global gui_instance
        gui_instance = ConsoleLogger()
        cli.run_app(WorkerOptions(entrypoint_fnc=console_entrypoint))
    else:
        # GUI mode
        root = tk.Tk()
        app = VoiceAgentGUI(root)

        def on_closing():
            async_runner.stop()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()


if __name__ == "__main__":
    main()
