#!/usr/bin/env python3
"""
Voice Agent Test Application - Enhanced Tkinter GUI
A voice agent demo with toggleable features for testing clean tool execution.
"""

import asyncio
import json
import os
import random
import subprocess
import threading
import time
import wave
import math
import tempfile
from collections import deque
from datetime import datetime
from typing import Any, AsyncIterable

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
    tts as lk_tts,
    stt as lk_stt,
    utils,
    APIConnectOptions,
)
from livekit.plugins import deepgram, openai, cartesia, silero

# Try to import pyttsx3 for local TTS
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# Try to import whisper for local STT
try:
    import whisper
    import numpy as np
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Load environment variables
load_dotenv()


# ============================================================================
# LOCAL TTS ENGINES
# ============================================================================

class LocalMacOSTTS(lk_tts.TTS):
    """Local TTS using macOS built-in 'say' command."""

    def __init__(self, voice: str = "Samantha", rate: int = 175):
        super().__init__(
            capabilities=lk_tts.TTSCapabilities(streaming=False),
            sample_rate=22050,
            num_channels=1,
        )
        self.voice = voice
        self.rate = rate

    def synthesize(self, text: str) -> lk_tts.ChunkedStream:
        return LocalMacOSChunkedStream(text, self.voice, self.rate, self)


class LocalMacOSChunkedStream(lk_tts.ChunkedStream):
    """Chunked stream for macOS TTS."""

    def __init__(self, text: str, voice: str, rate: int, tts: LocalMacOSTTS):
        super().__init__(tts=tts, input_text=text)
        self._text = text
        self._voice = voice
        self._rate = rate

    async def _run(self) -> None:
        try:
            with tempfile.NamedTemporaryFile(suffix='.aiff', delete=False) as f:
                temp_path = f.name

            # Use macOS say command to generate audio
            cmd = [
                'say',
                '-v', self._voice,
                '-r', str(self._rate),
                '-o', temp_path,
                '--data-format=LEI16@22050',
                self._text
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()

            if process.returncode == 0 and os.path.exists(temp_path):
                with open(temp_path, 'rb') as f:
                    audio_data = f.read()

                # Skip AIFF header (varies, but typically around 54 bytes)
                # For raw PCM we'd need to parse properly, but for simplicity
                # we'll send chunks
                chunk_size = 4096
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    self._event_ch.send_nowait(
                        lk_tts.SynthesizedAudio(
                            request_id=self._input_text,
                            frame=lk_tts.AudioFrame(
                                data=chunk,
                                sample_rate=22050,
                                num_channels=1,
                                samples_per_channel=len(chunk) // 2,
                            )
                        )
                    )

                os.unlink(temp_path)

        except Exception as e:
            print(f"[LocalTTS] Error: {e}")


class LocalPyttsx3TTS(lk_tts.TTS):
    """Local TTS using pyttsx3 (cross-platform)."""

    def __init__(self, voice_id: str | None = None, rate: int = 175):
        super().__init__(
            capabilities=lk_tts.TTSCapabilities(streaming=False),
            sample_rate=22050,
            num_channels=1,
        )
        self.voice_id = voice_id
        self.rate = rate
        self._engine = None
        if PYTTSX3_AVAILABLE:
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', rate)
            if voice_id:
                self._engine.setProperty('voice', voice_id)

    def synthesize(self, text: str) -> lk_tts.ChunkedStream:
        return LocalPyttsx3ChunkedStream(text, self._engine, self)


class LocalPyttsx3ChunkedStream(lk_tts.ChunkedStream):
    """Chunked stream for pyttsx3 TTS."""

    def __init__(self, text: str, engine, tts: LocalPyttsx3TTS):
        super().__init__(tts=tts, input_text=text)
        self._text = text
        self._engine = engine

    async def _run(self) -> None:
        if not PYTTSX3_AVAILABLE or not self._engine:
            return

        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name

            # Run pyttsx3 in a thread to avoid blocking
            def _synthesize():
                self._engine.save_to_file(self._text, temp_path)
                self._engine.runAndWait()

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _synthesize)

            if os.path.exists(temp_path):
                with wave.open(temp_path, 'rb') as wav_file:
                    audio_data = wav_file.readframes(wav_file.getnframes())
                    sample_rate = wav_file.getframerate()
                    num_channels = wav_file.getnchannels()

                chunk_size = 4096
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    self._event_ch.send_nowait(
                        lk_tts.SynthesizedAudio(
                            request_id=self._input_text,
                            frame=lk_tts.AudioFrame(
                                data=chunk,
                                sample_rate=sample_rate,
                                num_channels=num_channels,
                                samples_per_channel=len(chunk) // (2 * num_channels),
                            )
                        )
                    )

                os.unlink(temp_path)

        except Exception as e:
            print(f"[LocalTTS] pyttsx3 error: {e}")


def get_tts_provider(provider_str: str):
    """Get TTS provider based on configuration string."""
    provider, _, model = provider_str.partition('/')

    if provider == "local":
        if model == "macos" or model == "say":
            return LocalMacOSTTS(voice="Samantha", rate=175)
        elif model == "pyttsx3":
            if PYTTSX3_AVAILABLE:
                return LocalPyttsx3TTS(rate=175)
            else:
                print("[TTS] pyttsx3 not available, falling back to macOS say")
                return LocalMacOSTTS(voice="Samantha", rate=175)
        else:
            # Default to macOS on Darwin, pyttsx3 otherwise
            import platform
            if platform.system() == "Darwin":
                return LocalMacOSTTS(voice="Samantha", rate=175)
            elif PYTTSX3_AVAILABLE:
                return LocalPyttsx3TTS(rate=175)
            else:
                raise ValueError("No local TTS available. Install pyttsx3: pip install pyttsx3")

    elif provider == "cartesia":
        return cartesia.TTS(model=model or "sonic-english")

    elif provider == "elevenlabs":
        from livekit.plugins import elevenlabs
        return elevenlabs.TTS(model=model or "turbo-2")

    else:
        raise ValueError(f"Unknown TTS provider: {provider}")


# ============================================================================
# LOCAL STT ENGINES
# ============================================================================

class LocalWhisperSTT(lk_stt.STT):
    """Local STT using OpenAI's Whisper model."""

    def __init__(self, model_name: str = "base", sample_rate: int = 16000):
        super().__init__(
            capabilities=lk_stt.STTCapabilities(
                streaming=False,  # We simulate streaming with buffering
                interim_results=False,
            )
        )
        self.model_name = model_name
        self._sample_rate = sample_rate
        self._model = None

        if WHISPER_AVAILABLE:
            print(f"[Whisper] Loading model '{model_name}'...")
            self._model = whisper.load_model(model_name)
            print(f"[Whisper] Model loaded successfully")
        else:
            print("[Whisper] Whisper not available. Install with: pip install openai-whisper")

    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = APIConnectOptions(),
    ) -> "LocalWhisperStream":
        return LocalWhisperStream(
            stt=self,
            model=self._model,
            language=language,
            sample_rate=self._sample_rate,
        )


class LocalWhisperStream(lk_stt.SpeechStream):
    """Speech stream for local Whisper STT."""

    def __init__(
        self,
        *,
        stt: LocalWhisperSTT,
        model,
        language: str | None,
        sample_rate: int,
    ):
        super().__init__(stt=stt, sample_rate=sample_rate)
        self._model = model
        self._language = language
        self._audio_buffer = bytearray()
        self._closed = False

    async def _run(self) -> None:
        """Main processing loop - buffers audio and transcribes on flush."""
        from livekit import rtc

        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._sample_rate,
            num_channels=1,
            samples_per_channel=self._sample_rate // 100,  # 10ms chunks
        )

        async for data in self._input_ch:
            if isinstance(data, rtc.AudioFrame):
                # Accumulate audio data
                self._audio_buffer.extend(data.data.tobytes())
            elif isinstance(data, self._FlushSentinel):
                # Flush signal - transcribe the buffered audio
                if len(self._audio_buffer) > 0 and self._model:
                    await self._transcribe_buffer()
                # Clear buffer after transcription
                self._audio_buffer.clear()

    async def _transcribe_buffer(self) -> None:
        """Transcribe the accumulated audio buffer using Whisper."""
        if not self._model or len(self._audio_buffer) == 0:
            return

        try:
            # Convert bytes to numpy array for Whisper
            audio_data = np.frombuffer(self._audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

            # Ensure we have enough audio (Whisper needs at least 30ms typically)
            if len(audio_data) < 480:  # 30ms at 16kHz
                return

            # Run Whisper transcription in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(audio_data, language=self._language, fp16=False)
            )

            text = result.get("text", "").strip()
            detected_language = result.get("language", self._language or "en")

            if text:  # Only send event if we got actual text
                event = lk_stt.SpeechEvent(
                    type=lk_stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        lk_stt.SpeechData(
                            text=text,
                            language=detected_language,
                            confidence=1.0,  # Whisper doesn't provide confidence scores
                        )
                    ],
                )
                self._event_ch.send_nowait(event)

        except Exception as e:
            print(f"[Whisper] Transcription error: {e}")


def get_stt_provider(provider_str: str):
    """Get STT provider based on configuration string."""
    provider, _, model = provider_str.partition('/')

    if provider == "local":
        if not WHISPER_AVAILABLE:
            print("[STT] Whisper not available. Install with: pip install openai-whisper")
            print("[STT] Falling back to Deepgram")
            return deepgram.STT(model="nova-2")

        model = model or "base"
        print(f"[STT] Using local Whisper model: {model}")
        return LocalWhisperSTT(model_name=model)

    elif provider == "deepgram":
        return deepgram.STT(model=model or "nova-2")

    else:
        raise ValueError(f"Unknown STT provider: {provider}")


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

AUTO_GREETING_PROMPT = """
When the session starts, greet the user warmly and ask how you can help them today.
Keep it brief and friendly.
"""


# ============================================================================
# GUI STATE (shared with tools)
# ============================================================================

class DemoFeatures:
    """Toggleable demo features."""
    def __init__(self):
        self.verbose_tool_logging = False
        self.auto_greeting = False
        self.simulated_latency = False
        self.latency_ms = 1000
        self.sound_effects = False
        self.tool_timeline = False
        self.transcript_mode = False
        self.debug_mode = False
        self.announcement_mode = False
        self.conversation_summary = True
        self.audio_level_meter = True
        self.mock_mode = False

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def from_dict(self, data):
        for k, v in data.items():
            if hasattr(self, k):
                setattr(self, k, v)


demo_features = DemoFeatures()
gui_instance = None


# ============================================================================
# AUDIO LEVEL METER
# ============================================================================

class AudioLevelMeter:
    """Simulated audio level meter for visual feedback."""

    def __init__(self, canvas, width, height):
        self.canvas = canvas
        self.width = width
        self.height = height
        self.bars = []
        self.bar_count = 20
        self.bar_width = width / self.bar_count - 2
        self.max_height = height - 10

        # Create bars
        for i in range(self.bar_count):
            x = i * (self.bar_width + 2) + 1
            bar = canvas.create_rectangle(
                x, height - 5, x + self.bar_width, height,
                fill="#2a4a6e", outline=""
            )
            self.bars.append(bar)

        self.is_running = False
        self.animation_task = None

    def start(self):
        self.is_running = True
        self._animate()

    def stop(self):
        self.is_running = False
        # Reset all bars
        for bar in self.bars:
            self.canvas.coords(bar, 0, self.height, 0, self.height)

    def _animate(self):
        if not self.is_running:
            return

        # Simulate audio level based on current state
        if gui_instance and gui_instance.current_status == 'speaking':
            level = random.uniform(0.3, 0.9)
        elif gui_instance and gui_instance.current_status == 'listening':
            level = random.uniform(0.0, 0.2)
        else:
            level = random.uniform(0.0, 0.05)

        active_bars = int(level * self.bar_count)

        for i, bar in enumerate(self.bars):
            if i < active_bars:
                # Color gradient from green to red
                intensity = i / self.bar_count
                if intensity < 0.5:
                    color = f"#{int(0 + intensity*2*100):02x}ff88"
                else:
                    color = f"#ff{int(255 - (intensity-0.5)*2*100):02x}88"

                x = i * (self.bar_width + 2) + 1
                h = int(intensity * self.max_height)
                self.canvas.coords(bar, x, self.height - h, x + self.bar_width, self.height)
                self.canvas.itemconfig(bar, fill=color)
            else:
                x = i * (self.bar_width + 2) + 1
                self.canvas.coords(bar, x, self.height - 2, x + self.bar_width, self.height)
                self.canvas.itemconfig(bar, fill="#2a4a6e")

        self.canvas.after(50, self._animate)


# ============================================================================
# TOOL TIMELINE
# ============================================================================

class ToolTimeline:
    """Visual timeline of tool executions."""

    def __init__(self, canvas, width, height):
        self.canvas = canvas
        self.width = width
        self.height = height
        self.entries = []
        self.start_time = time.time()

        # Draw background grid
        for i in range(0, width, 50):
            canvas.create_line(i, 0, i, height, fill="#1a1a3e", width=1)

    def add_tool_call(self, tool_name, duration_ms):
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Scale x position (10 seconds = full width)
        x = min((elapsed / 10) * self.width, self.width - 60)

        # Color by duration
        if duration_ms < 500:
            color = "#00ff88"  # Green - quick
        elif duration_ms < 2000:
            color = "#ffd93d"  # Yellow - medium
        elif duration_ms < 5000:
            color = "#ffa500"  # Orange - long
        else:
            color = "#ff4757"  # Red - very long

        # Draw marker
        y = 15
        self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill=color, outline="white", width=1)

        # Add tooltip text
        text = f"{tool_name}: {duration_ms}ms"
        self.canvas.create_text(x, y + 20, text=text, fill="white", font=("Arial", 8), angle=45)

    def reset(self):
        self.canvas.delete("all")
        self.start_time = time.time()
        self.entries = []

        # Redraw grid
        for i in range(0, self.width, 50):
            self.canvas.create_line(i, 0, i, self.height, fill="#1a1a3e", width=1)


# ============================================================================
# TEST TOOLS
# ============================================================================

@function_tool
async def get_current_time(context: RunContext[AgentSession], timezone: str = "UTC") -> str:
    """Returns the current time in the user's timezone."""
    start_time = time.time()

    # Apply simulated latency if enabled
    if demo_features.simulated_latency:
        await asyncio.sleep(demo_features.latency_ms / 1000)

    result = datetime.now().astimezone().strftime("%I:%M %p")
    elapsed = (time.time() - start_time) * 1000

    if gui_instance:
        gui_instance.log_tool_execution("get_current_time", timezone, result, elapsed)
        if demo_features.tool_timeline:
            gui_instance.timeline.add_tool_call("get_current_time", elapsed)

    return f"The current time is {result}."


@function_tool
async def flip_coin(context: RunContext[AgentSession]) -> str:
    """Flips a coin and returns heads or tails."""
    start_time = time.time()

    if demo_features.simulated_latency:
        await asyncio.sleep(demo_features.latency_ms / 1000)

    result = random.choice(["heads", "tails"])
    elapsed = (time.time() - start_time) * 1000

    if gui_instance:
        gui_instance.log_tool_execution("flip_coin", None, result, elapsed)
        if demo_features.tool_timeline:
            gui_instance.timeline.add_tool_call("flip_coin", elapsed)

    return f"It's {result}."


@function_tool
async def roll_dice(context: RunContext[AgentSession], sides: int = 6) -> str:
    """Rolls a die with the specified number of sides (default 6)."""
    start_time = time.time()

    if demo_features.simulated_latency:
        await asyncio.sleep(demo_features.latency_ms / 1000)

    if sides < 2 or sides > 100:
        sides = 6
    result = random.randint(1, sides)
    elapsed = (time.time() - start_time) * 1000

    if gui_instance:
        gui_instance.log_tool_execution("roll_dice", f"sides={sides}", result, elapsed)
        if demo_features.tool_timeline:
            gui_instance.timeline.add_tool_call("roll_dice", elapsed)

    return f"You rolled a {result}."


@function_tool
async def get_weather(context: RunContext[AgentSession], location: str) -> str:
    """Gets the current weather for a location."""
    start_time = time.time()
    await asyncio.sleep(1.5)

    if demo_features.simulated_latency:
        await asyncio.sleep(demo_features.latency_ms / 1000)

    conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "windy"]
    temp = random.randint(45, 95)
    condition = random.choice(conditions)
    elapsed = (time.time() - start_time) * 1000

    if gui_instance:
        gui_instance.log_tool_execution("get_weather", location, f"{temp}°F {condition}", elapsed)
        if demo_features.tool_timeline:
            gui_instance.timeline.add_tool_call("get_weather", elapsed)

    return f"It's currently {temp} degrees Fahrenheit and {condition} in {location}."


@function_tool
async def calculator(context: RunContext[AgentSession], expression: str) -> str:
    """Evaluates a mathematical expression."""
    start_time = time.time()

    if demo_features.simulated_latency:
        await asyncio.sleep(demo_features.latency_ms / 1000)

    try:
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names)
        elapsed = (time.time() - start_time) * 1000

        if gui_instance:
            gui_instance.log_tool_execution("calculator", expression, str(result), elapsed)
            if demo_features.tool_timeline:
                gui_instance.timeline.add_tool_call("calculator", elapsed)

        return f"The result is {result}."
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        if gui_instance:
            gui_instance.log_tool_execution("calculator", expression, f"ERROR: {e}", elapsed, is_error=True)
        return f"Sorry, I couldn't calculate that. The error was: {str(e)}"


@function_tool
async def web_search(context: RunContext[AgentSession], query: str) -> str:
    """Searches the web for information."""
    start_time = time.time()
    await asyncio.sleep(2.0)

    if demo_features.simulated_latency:
        await asyncio.sleep(demo_features.latency_ms / 1000)

    results = [
        f"Here's what I found about '{query}':",
        f"1. Wikipedia: {query} is a topic of interest with many applications.",
        f"2. News: Recent updates about {query} have been trending.",
    ]
    elapsed = (time.time() - start_time) * 1000

    if gui_instance:
        gui_instance.log_tool_execution("web_search", query, f"{len(results)} results", elapsed)
        if demo_features.tool_timeline:
            gui_instance.timeline.add_tool_call("web_search", elapsed)

    return "\n".join(results)


@function_tool
async def analyze_text(context: RunContext[AgentSession], text: str, analysis_type: str = "sentiment") -> str:
    """Analyzes text using AI. Types: sentiment, summary, keywords, translation."""
    start_time = time.time()
    await asyncio.sleep(4.0)

    if demo_features.simulated_latency:
        await asyncio.sleep(demo_features.latency_ms / 1000)

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
        gui_instance.log_tool_execution("analyze_text", f"{text_preview} ({analysis_type})", result.split(':')[1] if ':' in result else result, elapsed)
        if demo_features.tool_timeline:
            gui_instance.timeline.add_tool_call("analyze_text", elapsed)

    return result


@function_tool
async def generate_report(context: RunContext[AgentSession], topic: str, sections: list[str], detail_level: str = "brief") -> str:
    """Generates a detailed report on a topic."""
    start_time = time.time()
    await asyncio.sleep(6.0)

    if demo_features.simulated_latency:
        await asyncio.sleep(demo_features.latency_ms / 1000)

    section_names = ", ".join(sections)
    result = f"""Report on {topic}:
Sections: {section_names}
Detail Level: {detail_level}

{' '.join([f"{s.capitalize()}: Analysis and findings for this section." for s in sections[:3]])}

Overall conclusion: The report highlights important aspects of {topic}."""

    elapsed = (time.time() - start_time) * 1000

    if gui_instance:
        gui_instance.log_tool_execution("generate_report", f"{topic} ({len(sections)} sections)", "Complete", elapsed)
        if demo_features.tool_timeline:
            gui_instance.timeline.add_tool_call("generate_report", elapsed)

    return result


@function_tool
async def database_query(context: RunContext[AgentSession], sql: str, filters: dict[str, Any] | None = None) -> str:
    """Executes a database query with optional filters."""
    start_time = time.time()
    await asyncio.sleep(5.0)

    if demo_features.simulated_latency:
        await asyncio.sleep(demo_features.latency_ms / 1000)

    filters_str = json.dumps(filters) if filters else "none"
    result_count = random.randint(5, 50)
    result = f"Query executed: {sql}\nFilters: {filters_str}\nResults: {result_count} records found."

    elapsed = (time.time() - start_time) * 1000

    if gui_instance:
        gui_instance.log_tool_execution("database_query", sql, f"{result_count} records", elapsed)
        if demo_features.tool_timeline:
            gui_instance.timeline.add_tool_call("database_query", elapsed)

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
        gui_instance.log_tool_execution("trigger_error", error_type, f"Simulated: {error_type}", elapsed, is_error=True)
        if demo_features.tool_timeline:
            gui_instance.timeline.add_tool_call("trigger_error", elapsed)

    return f"I encountered an issue: {error_messages.get(error_type, 'Unknown error occurred')}. Please try again."


# ============================================================================
# ADDITIONAL DEMO TOOLS
# ============================================================================

@function_tool
async def get_system_info(context: RunContext[AgentSession]) -> str:
    """Returns system information for debugging."""
    import platform
    info = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "session_active": gui_instance.is_running if gui_instance else False,
    }
    return f"System info: {json.dumps(info, indent=2)}"


@function_tool
async def set_agent_mood(context: RunContext[AgentSession], mood: str) -> str:
    """Sets the agent's mood for the conversation. Moods: friendly, professional, playful, terse."""
    valid_moods = ["friendly", "professional", "playful", "terse"]
    if mood.lower() not in valid_moods:
        return f"Please choose a valid mood: {', '.join(valid_moods)}"

    if gui_instance:
        gui_instance.agent_mood = mood.lower()
        gui_instance.log("System", f"Agent mood set to: {mood}", "system")

    return f"I'll now respond in a more {mood} manner."


@function_tool
async def get_session_stats(context: RunContext[AgentSession]) -> str:
    """Returns statistics about the current session."""
    if gui_instance:
        stats = gui_instance.get_session_stats()
        return f"Session stats: {json.dumps(stats, indent=2)}"
    return "No session statistics available."


# ============================================================================
# VOICE AGENT
# ============================================================================

class VoiceAgentTest(Agent):
    """A voice agent for testing clean tool execution."""

    def __init__(self, llm_provider: str = "openai/gpt-4o-mini"):
        # Build instructions based on demo features
        instructions = ""

        if demo_features.announcement_mode:
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

Current mood: {getattr(self, 'mood', 'friendly')}
"""

        if demo_features.debug_mode:
            instructions += "\n\nDEBUG MODE: You can share internal reasoning when asked."

        super().__init__(instructions=instructions)
        self.mood = "friendly"


# ============================================================================
# ASYNCIO RUNNER
# ============================================================================

class AsyncioRunner:
    """Runs asyncio event loop in a separate thread."""

    def __init__(self):
        self.loop = None
        self.thread = None
        self.running = False

    def start(self):
        self.loop = asyncio.new_event_loop()
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def stop(self):
        self.running = False
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=2)

    def run_coroutine(self, coro):
        if self.loop:
            return asyncio.run_coroutine_threadsafe(coro, self.loop)
        return None


async_runner = AsyncioRunner()


# ============================================================================
# ENHANCED TKINTER GUI
# ============================================================================

class VoiceAgentGUI:
    """Enhanced GUI application for Voice Agent Test."""

    # Checkboxes configuration
    CHECKBOXES = [
        ("verbose_tool_logging", "Verbose Tool Logging", "Show detailed tool execution info"),
        ("auto_greeting", "Auto Greeting", "Agent greets on session start"),
        ("simulated_latency", "Simulated Latency", "Add artificial delay to responses"),
        ("sound_effects", "Sound Effects", "Play sounds on tool execution"),
        ("tool_timeline", "Tool Timeline", "Visual timeline of tool calls"),
        ("transcript_mode", "Transcript Mode", "Show real-time transcription"),
        ("debug_mode", "Debug Mode", "Show internal agent state"),
        ("announcement_mode", "Announcement Mode", "Agent announces tools (for comparison)"),
        ("conversation_summary", "Conversation Summary", "Show stats at session end"),
        ("audio_level_meter", "Audio Level Meter", "Visual audio input indicator"),
        ("mock_mode", "Mock Mode", "Use fake responses (faster testing)"),
    ]

    def __init__(self, root):
        global gui_instance
        gui_instance = self

        self.root = root
        self.root.title("Voice Agent Test - Enhanced Demo")
        self.root.geometry("1100x850")
        self.root.configure(bg="#0d0d1a")

        # State
        self.is_running = False
        self.session = None
        self.current_status = "disconnected"  # disconnected, connecting, listening, processing, speaking
        self.agent_mood = "friendly"
        self.session_start_time = None
        self.tool_calls = []
        self.user_messages = []
        self.agent_responses = []

        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Colors
        self.colors = {
            'bg': '#0d0d1a',
            'card_bg': '#1a1a2e',
            'card_bg_alt': '#252547',
            'text': '#ffffff',
            'accent': '#4a9eff',
            'accent2': '#7c4dff',
            'success': '#00ff88',
            'warning': '#ffa500',
            'danger': '#ff4757',
            'muted': '#666688',
        }

        # Build UI
        self._build_header()
        self._build_main_content()

        # Create components
        self.timeline = ToolTimeline(self.timeline_canvas, 500, 60)
        self.audio_meter = AudioLevelMeter(self.audio_canvas, 200, 40)

        # Load saved config
        self._load_config()

        # Start asyncio runner
        async_runner.start()

        # Start UI update loop
        self._update_ui()

    def _build_header(self):
        """Build the header section."""
        header = tk.Frame(self.root, bg="#0d0d1a", pady=8)
        header.pack(fill='x')

        title_row = tk.Frame(header, bg="#0d0d1a")
        title_row.pack()

        tk.Label(
            title_row,
            text="🎙️",
            font=("Arial", 28),
            bg="#0d0d1a",
            fg="#4a9eff"
        ).pack(side='left', padx=5)

        tk.Label(
            title_row,
            text="Voice Agent Test",
            font=("Arial", 22, "bold"),
            bg="#0d0d1a",
            fg="#ffffff"
        ).pack(side='left')

        tk.Label(
            title_row,
            text="Enhanced Demo",
            font=("Arial", 12),
            bg="#0d0d1a",
            fg=self.colors['accent2']
        ).pack(side='left', padx=10)

        tk.Label(
            header,
            text="Clean Tool Execution Demonstration with Toggleable Features",
            font=("Arial", 10),
            bg="#0d0d1a",
            fg="#666688"
        ).pack()

    def _build_main_content(self):
        """Build the main content area."""
        main = tk.Frame(self.root, bg="#0d0d1a")
        main.pack(fill='both', expand=True, padx=10, pady=5)

        # Left panel - Configuration and Features
        left_panel = tk.Frame(main, bg="#0d0d1a")
        left_panel.pack(side='left', fill='both', expand=True)

        self._build_config_panel(left_panel)
        self._build_features_panel(left_panel)
        self._build_status_panel(left_panel)
        self._build_visualization_panel(left_panel)

        # Right panel - Log and Stats
        right_panel = tk.Frame(main, bg="#0d0d1a", width=400)
        right_panel.pack(side='right', fill='both', expand=True)
        right_panel.pack_propagate(False)

        self._build_stats_panel(right_panel)
        self._build_log_panel(right_panel)
        self._build_quick_actions(right_panel)

    def _build_config_panel(self, parent):
        """Build the configuration panel."""
        config_frame = tk.Frame(parent, bg=self.colors['card_bg'], padx=15, pady=12)
        config_frame.pack(fill='x', padx=(0, 5), pady=5)

        # Collapsible header
        header = tk.Frame(config_frame, bg=self.colors['card_bg'])
        header.pack(fill='x')

        tk.Label(
            header,
            text="🔧 Configuration",
            font=("Arial", 11, "bold"),
            bg=self.colors['card_bg'],
            fg=self.colors['accent']
        ).pack(side='left')

        self.config_visible = True
        toggle_btn = tk.Label(
            header,
            text="−",
            font=("Arial", 12, "bold"),
            bg=self.colors['card_bg'],
            fg=self.colors['muted'],
            cursor="hand2"
        )
        toggle_btn.pack(side='right')
        toggle_btn.bind("<Button-1>", lambda e: self._toggle_section("config"))

        self.config_content = tk.Frame(config_frame, bg=self.colors['card_bg'])
        self.config_content.pack(fill='x', pady=10)

        # LiveKit settings (compact)
        lk_frame = tk.Frame(self.config_content, bg=self.colors['card_bg'])
        lk_frame.pack(fill='x', pady=3)

        tk.Label(lk_frame, text="URL:", bg=self.colors['card_bg'], fg=self.colors['muted'], width=5, anchor='w').grid(row=0, column=0, sticky='w')
        self.livekit_url = tk.Entry(lk_frame, bg="#2a2a4e", fg="white", insertbackground='white', font=("Arial", 9))
        self.livekit_url.grid(row=0, column=1, sticky='ew', padx=2)

        tk.Label(lk_frame, text="Key:", bg=self.colors['card_bg'], fg=self.colors['muted'], width=4, anchor='w').grid(row=0, column=2, sticky='w')
        self.api_key = tk.Entry(lk_frame, bg="#2a2a4e", fg="white", insertbackground='white', show='*', font=("Arial", 9))
        self.api_key.grid(row=0, column=3, sticky='ew', padx=2)

        tk.Label(lk_frame, text="Secret:", bg=self.colors['card_bg'], fg=self.colors['muted'], width=6, anchor='w').grid(row=0, column=4, sticky='w')
        self.api_secret = tk.Entry(lk_frame, bg="#2a2a4e", fg="white", insertbackground='white', show='*', font=("Arial", 9))
        self.api_secret.grid(row=0, column=5, sticky='ew', padx=2)

        tk.Label(lk_frame, text="Room:", bg=self.colors['card_bg'], fg=self.colors['muted'], width=5, anchor='w').grid(row=1, column=0, sticky='w')
        self.room_name = tk.Entry(lk_frame, bg="#2a2a4e", fg="white", insertbackground='white', font=("Arial", 9))
        self.room_name.grid(row=1, column=1, sticky='ew', padx=2)

        lk_frame.columnconfigure(1, weight=1)
        lk_frame.columnconfigure(3, weight=1)
        lk_frame.columnconfigure(5, weight=1)

        # Provider settings
        provider_frame = tk.Frame(self.config_content, bg=self.colors['card_bg'])
        provider_frame.pack(fill='x', pady=3)

        tk.Label(provider_frame, text="LLM:", bg=self.colors['card_bg'], fg=self.colors['muted'], width=4, anchor='w').grid(row=0, column=0, sticky='w')
        self.llm_provider = ttk.Combobox(provider_frame, values=[
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "openai/o3-mini",
            "anthropic/claude-3-haiku",
            "anthropic/claude-3.5-sonnet",
        ], state='readonly', font=("Arial", 9))
        self.llm_provider.set("openai/gpt-4o-mini")
        self.llm_provider.grid(row=0, column=1, sticky='ew', padx=2)

        tk.Label(provider_frame, text="STT:", bg=self.colors['card_bg'], fg=self.colors['muted'], width=4, anchor='w').grid(row=0, column=2, sticky='w')
        self.stt_provider = ttk.Combobox(provider_frame, values=[
            "local/whisper-tiny",
            "local/whisper-base",
            "local/whisper-small",
            "local/whisper-medium",
            "deepgram/nova-2",
            "deepgram/nova-3",
        ], state='readonly', font=("Arial", 9))
        self.stt_provider.set("local/whisper-base")
        self.stt_provider.grid(row=0, column=3, sticky='ew', padx=2)

        tk.Label(provider_frame, text="TTS:", bg=self.colors['card_bg'], fg=self.colors['muted'], width=4, anchor='w').grid(row=0, column=4, sticky='w')
        self.tts_provider = ttk.Combobox(provider_frame, values=[
            "local/macos",
            "local/pyttsx3",
            "cartesia/sonic-english",
            "cartesia/sonic-2",
            "elevenlabs/turbo-2",
        ], state='readonly', font=("Arial", 9))
        self.tts_provider.set("local/macos")
        self.tts_provider.grid(row=0, column=5, sticky='ew', padx=2)

        provider_frame.columnconfigure(1, weight=1)
        provider_frame.columnconfigure(3, weight=1)
        provider_frame.columnconfigure(5, weight=1)

    def _build_features_panel(self, parent):
        """Build the features panel with checkboxes."""
        features_frame = tk.Frame(parent, bg=self.colors['card_bg_alt'], padx=15, pady=12)
        features_frame.pack(fill='x', padx=(0, 5), pady=5)

        # Header
        header = tk.Frame(features_frame, bg=self.colors['card_bg_alt'])
        header.pack(fill='x')

        tk.Label(
            header,
            text="⚙️ Demo Features",
            font=("Arial", 11, "bold"),
            bg=self.colors['card_bg_alt'],
            fg=self.colors['accent2']
        ).pack(side='left')

        self.feature_vars = {}

        # Checkboxes grid
        checkbox_frame = tk.Frame(features_frame, bg=self.colors['card_bg_alt'])
        checkbox_frame.pack(fill='x', pady=10)

        for i, (attr, label, tooltip) in enumerate(self.CHECKBOXES):
            var = tk.BooleanVar(value=getattr(demo_features, attr, False))
            self.feature_vars[attr] = var

            row = i // 2
            col = i % 2

            cb = tk.Checkbutton(
                checkbox_frame,
                text=label,
                variable=var,
                bg=self.colors['card_bg_alt'],
                fg=self.colors['text'],
                selectcolor=self.colors['card_bg'],
                activebackground=self.colors['card_bg_alt'],
                activeforeground=self.colors['accent'],
                cursor="hand2",
                font=("Arial", 9),
                command=lambda a=attr, v=var: self._toggle_feature(a, v.get())
            )
            cb.grid(row=row, column=col, sticky='w', padx=5, pady=2)

            # Store reference for tooltip
            cb.tooltip_text = tooltip

        # Latency slider (only enabled when simulated_latency is on)
        latency_frame = tk.Frame(features_frame, bg=self.colors['card_bg_alt'])
        latency_frame.pack(fill='x', pady=5)

        tk.Label(
            latency_frame,
            text="Latency:",
            bg=self.colors['card_bg_alt'],
            fg=self.colors['muted'],
            font=("Arial", 9)
        ).pack(side='left')

        self.latency_slider = tk.Scale(
            latency_frame,
            from_=0,
            to=3000,
            orient='horizontal',
            bg=self.colors['card_bg_alt'],
            fg=self.colors['text'],
            highlightthickness=0,
            length=200,
            showvalue=0,
            command=self._update_latency
        )
        self.latency_slider.set(1000)
        self.latency_slider.pack(side='left', padx=5)
        self.latency_slider.config(state='disabled')

        self.latency_label = tk.Label(
            latency_frame,
            text="1000ms",
            bg=self.colors['card_bg_alt'],
            fg=self.colors['muted'],
            font=("Arial", 9),
            width=6
        )
        self.latency_label.pack(side='left')

    def _build_status_panel(self, parent):
        """Build the status and control panel."""
        status_frame = tk.Frame(parent, bg=self.colors['card_bg'], padx=15, pady=12)
        status_frame.pack(fill='x', padx=(0, 5), pady=5)

        tk.Label(
            status_frame,
            text="📊 Session Status",
            font=("Arial", 11, "bold"),
            bg=self.colors['card_bg'],
            fg=self.colors['accent']
        ).pack(anchor='w')

        # Status row
        status_row = tk.Frame(status_frame, bg=self.colors['card_bg'])
        status_row.pack(pady=8)

        # Status dot
        self.status_dot = tk.Canvas(
            status_row,
            width=24,
            height=24,
            bg=self.colors['card_bg'],
            highlightthickness=0
        )
        self.status_dot.pack(side='left', padx=10)
        self._draw_status_dot('#555555')

        # Status text
        self.status_text = tk.Label(
            status_row,
            text="Disconnected",
            font=("Arial", 13, "bold"),
            bg=self.colors['card_bg'],
            fg="#ffffff"
        )
        self.status_text.pack(side='left')

        # Agent mood indicator
        self.mood_label = tk.Label(
            status_row,
            text="😊 friendly",
            font=("Arial", 10),
            bg=self.colors['card_bg'],
            fg=self.colors['muted']
        )
        self.mood_label.pack(side='left', padx=20)

        # Control buttons
        button_frame = tk.Frame(status_frame, bg=self.colors['card_bg'])
        button_frame.pack(pady=8)

        self.start_btn = tk.Button(
            button_frame,
            text="▶ START",
            font=("Arial", 13, "bold"),
            bg=self.colors['success'],
            fg="#0d0d1a",
            width=12,
            height=1,
            cursor="hand2",
            command=self.start_session,
            relief='flat',
            bd=0
        )
        self.start_btn.pack(side='left', padx=8)

        self.stop_btn = tk.Button(
            button_frame,
            text="■ STOP",
            font=("Arial", 14, "bold"),
            bg=self.colors['danger'],
            fg="white",
            width=12,
            height=1,
            cursor="hand2",
            command=self.stop_session,
            relief='flat',
            bd=0,
            state='disabled'
        )
        self.stop_btn.pack(side='left', padx=8)

    def _build_visualization_panel(self, parent):
        """Build visualization panels (timeline and audio meter)."""
        viz_frame = tk.Frame(parent, bg=self.colors['card_bg_alt'], padx=15, pady=12)
        viz_frame.pack(fill='x', padx=(0, 5), pady=5)

        # Timeline
        tk.Label(
            viz_frame,
            text="📈 Tool Timeline (last 10s)",
            font=("Arial", 9, "bold"),
            bg=self.colors['card_bg_alt'],
            fg=self.colors['muted']
        ).pack(anchor='w')

        self.timeline_canvas = tk.Canvas(
            viz_frame,
            width=500,
            height=60,
            bg="#0d0d1a",
            highlightthickness=1,
            highlightbackground=self.colors['card_bg']
        )
        self.timeline_canvas.pack(fill='x', pady=5)

        # Audio meter and timer
        meter_row = tk.Frame(viz_frame, bg=self.colors['card_bg_alt'])
        meter_row.pack(fill='x', pady=5)

        tk.Label(
            meter_row,
            text="🔊 Audio Level:",
            font=("Arial", 9),
            bg=self.colors['card_bg_alt'],
            fg=self.colors['muted']
        ).pack(side='left')

        self.audio_canvas = tk.Canvas(
            meter_row,
            width=200,
            height=40,
            bg="#0d0d1a",
            highlightthickness=1,
            highlightbackground=self.colors['card_bg']
        )
        self.audio_canvas.pack(side='left', padx=5)

        self.session_timer = tk.Label(
            meter_row,
            text="00:00",
            font=("Consolas", 14, "bold"),
            bg=self.colors['card_bg_alt'],
            fg=self.colors['accent']
        )
        self.session_timer.pack(side='right')

    def _build_stats_panel(self, parent):
        """Build the statistics panel."""
        stats_frame = tk.Frame(parent, bg=self.colors['card_bg'], padx=15, pady=12)
        stats_frame.pack(fill='x', pady=5)

        tk.Label(
            stats_frame,
            text="📊 Session Statistics",
            font=("Arial", 11, "bold"),
            bg=self.colors['card_bg'],
            fg=self.colors['accent']
        ).pack(anchor='w')

        # Stats grid
        self.stats_labels = {}
        stats_grid = tk.Frame(stats_frame, bg=self.colors['card_bg'])
        stats_grid.pack(fill='x', pady=8)

        stats = [
            ("duration", "Duration", "00:00"),
            ("tool_calls", "Tool Calls", "0"),
            ("avg_latency", "Avg Latency", "0ms"),
            ("user_msgs", "User Messages", "0"),
            ("agent_resp", "Agent Responses", "0"),
        ]

        for i, (key, label, default) in enumerate(stats):
            row = i // 3
            col = i % 3

            lbl_frame = tk.Frame(stats_grid, bg="#2a2a4e", padx=8, pady=5)
            lbl_frame.grid(row=row, column=col, padx=3, pady=3, sticky='ew')

            tk.Label(
                lbl_frame,
                text=label,
                font=("Arial", 8),
                bg="#2a2a4e",
                fg=self.colors['muted']
            ).pack()

            val_label = tk.Label(
                lbl_frame,
                text=default,
                font=("Arial", 12, "bold"),
                bg="#2a2a4e",
                fg=self.colors['accent']
            )
            val_label.pack()
            self.stats_labels[key] = val_label

        # Mood selection
        mood_frame = tk.Frame(stats_frame, bg=self.colors['card_bg'])
        mood_frame.pack(fill='x', pady=5)

        tk.Label(
            mood_frame,
            text="Agent Mood:",
            font=("Arial", 9),
            bg=self.colors['card_bg'],
            fg=self.colors['muted']
        ).pack(side='left')

        self.mood_var = tk.StringVar(value="friendly")
        moods = ["friendly", "professional", "playful", "terse"]
        for mood in moods:
            rb = tk.Radiobutton(
                mood_frame,
                text=mood.capitalize(),
                variable=self.mood_var,
                value=mood,
                bg=self.colors['card_bg'],
                fg=self.colors['text'],
                selectcolor="#2a2a4e",
                activebackground=self.colors['card_bg'],
                font=("Arial", 8),
                command=self._change_mood
            )
            rb.pack(side='left', padx=5)

    def _build_log_panel(self, parent):
        """Build the conversation log panel."""
        log_frame = tk.Frame(parent, bg=self.colors['card_bg'], padx=15, pady=12)
        log_frame.pack(fill='both', expand=True, pady=5)

        tk.Label(
            log_frame,
            text="💬 Conversation Log",
            font=("Arial", 11, "bold"),
            bg=self.colors['card_bg'],
            fg=self.colors['accent']
        ).pack(anchor='w')

        # Log container
        log_container = tk.Frame(log_frame, bg=self.colors['card_bg'])
        log_container.pack(fill='both', expand=True, pady=8)

        scrollbar = tk.Scrollbar(log_container, bg="#2a2a4e")
        scrollbar.pack(side='right', fill='y')

        self.log_text = tk.Text(
            log_container,
            bg="#0a0a14",
            fg="#00ff88",
            font=("Consolas", 9),
            yscrollcommand=scrollbar.set,
            insertbackground='white',
            relief='flat',
            padx=8,
            pady=8,
            wrap='word'
        )
        self.log_text.pack(fill='both', expand=True)
        scrollbar.config(command=self.log_text.yview)

        # Configure tags
        self.log_text.tag_config('system', foreground='#666688')
        self.log_text.tag_config('user', foreground='#4a9eff')
        self.log_text.tag_config('agent', foreground='#00ff88')
        self.log_text.tag_config('tool', foreground='#ff6b6b')
        self.log_text.tag_config('error', foreground='#ff4757')
        self.log_text.tag_config('debug', foreground='#ffd93d')
        self.log_text.tag_config('timestamp', foreground='#444466')

        self.log("System", "Ready. Configure settings and click START.", "system")

        # Clear and Export buttons
        btn_row = tk.Frame(log_frame, bg=self.colors['card_bg'])
        btn_row.pack(fill='x')

        tk.Button(
            btn_row,
            text="Clear",
            font=("Arial", 9),
            bg="#2a2a4e",
            fg="white",
            cursor="hand2",
            command=self.clear_log
        ).pack(side='left', padx=5)

        tk.Button(
            btn_row,
            text="Export",
            font=("Arial", 9),
            bg="#2a2a4e",
            fg="white",
            cursor="hand2",
            command=self.export_log
        ).pack(side='left', padx=5)

    def _build_quick_actions(self, parent):
        """Build quick action buttons."""
        actions_frame = tk.Frame(parent, bg=self.colors['card_bg_alt'], padx=15, pady=12)
        actions_frame.pack(fill='x', pady=5)

        tk.Label(
            actions_frame,
            text="⚡ Quick Test Actions",
            font=("Arial", 11, "bold"),
            bg=self.colors['card_bg_alt'],
            fg=self.colors['accent2']
        ).pack(anchor='w')

        buttons_frame = tk.Frame(actions_frame, bg=self.colors['card_bg_alt'])
        buttons_frame.pack(fill='x', pady=8)

        actions = [
            ("What time is it?", "🕐"),
            ("Flip a coin", "🪙"),
            ("Roll d20", "🎲"),
            ("Weather in Tokyo", "🌤️"),
            ("234 * 567", "🔢"),
            ("Search AI trends", "🔍"),
            ("Set mood: playful", "😄"),
            ("Get session stats", "📊"),
        ]

        for i, (label, emoji) in enumerate(actions):
            btn = tk.Button(
                buttons_frame,
                text=f"{emoji} {label}",
                font=("Arial", 8),
                bg="#2a2a4e",
                fg="white",
                cursor="hand2",
                relief='flat',
                command=lambda l=label: self.speak_test(l)
            )
            btn.grid(row=i//4, column=i%4, padx=3, pady=3, sticky='ew')

        for i in range(4):
            buttons_frame.columnconfigure(i, weight=1)

    # ============================================================================
    # UI HELPERS
    # ============================================================================

    def _toggle_section(self, section):
        if section == "config":
            self.config_visible = not self.config_visible
            if self.config_visible:
                self.config_content.pack(fill='x', pady=10)
            else:
                self.config_content.pack_forget()

    def _draw_status_dot(self, color):
        self.status_dot.delete("all")
        self.status_dot.create_oval(2, 2, 22, 22, fill=color, outline="")
        # Add glow effect
        self.status_dot.create_oval(4, 4, 20, 20, fill=color, outline="", stipple="gray50")

    def _toggle_feature(self, attr, value):
        setattr(demo_features, attr, value)

        # Special handling for simulated latency
        if attr == "simulated_latency":
            state = 'normal' if value else 'disabled'
            self.latency_slider.config(state=state)

        if attr == "audio_level_meter":
            if value and self.is_running:
                self.audio_meter.start()
            else:
                self.audio_meter.stop()

        if attr == "tool_timeline" and not value:
            self.timeline_canvas.delete("all")
            self.timeline.entries = []

        self.log("Debug", f"Feature '{attr}' set to {value}", "debug")

    def _update_latency(self, value):
        demo_features.latency_ms = int(value)
        self.latency_label.config(text=f"{int(value)}ms")

    def _change_mood(self):
        self.agent_mood = self.mood_var.get()
        mood_emojis = {"friendly": "😊", "professional": "👔", "playful": "😄", "terse": "😐"}
        self.mood_label.config(text=f"{mood_emojis.get(self.agent_mood, '😊')} {self.agent_mood}")

    def _update_ui(self):
        """Update UI elements periodically."""
        if self.is_running and self.session_start_time:
            # Update timer
            elapsed = int(time.time() - self.session_start_time)
            mins, secs = divmod(elapsed, 60)
            self.session_timer.config(text=f"{mins:02d}:{secs:02d}")
            self.stats_labels["duration"].config(text=f"{mins:02d}:{secs:02d}")

        self.root.after(100, self._update_ui)

    # ============================================================================
    # LOGGING
    # ============================================================================

    def log(self, source, message, tag='system'):
        if not demo_features.debug_mode and tag == 'debug':
            return

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_text.insert('end', f"[{timestamp}] ", 'timestamp')
        self.log_text.insert('end', f"{source}: ", tag)
        self.log_text.insert('end', f"{message}\n")
        self.log_text.see('end')

    def log_tool_execution(self, tool_name, args, result, duration_ms, is_error=False):
        """Log tool execution with configurable verbosity."""
        tag = 'error' if is_error else 'tool'

        if demo_features.verbose_tool_logging:
            args_str = str(args) if args else "no args"
            self.log("Tool", f"{tool_name}({args_str}) -> {result} ({duration_ms:.0f}ms)", tag)
        else:
            self.log("Tool", f"{tool_name} -> {duration_ms:.0f}ms", tag)

        # Update stats
        self.tool_calls.append({"tool": tool_name, "duration": duration_ms, "time": time.time()})
        avg_latency = sum(t["duration"] for t in self.tool_calls) / len(self.tool_calls) if self.tool_calls else 0
        self.stats_labels["tool_calls"].config(text=str(len(self.tool_calls)))
        self.stats_labels["avg_latency"].config(text=f"{avg_latency:.0f}ms")

        # Sound effect
        if demo_features.sound_effects and not is_error:
            self._play_sound(duration_ms)

    def clear_log(self):
        self.log_text.delete(1.0, 'end')

    def export_log(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                f.write(self.log_text.get(1.0, 'end'))

    def _play_sound(self, duration_ms):
        """Play a subtle sound based on tool duration."""
        # Visual feedback instead of audio for simplicity
        pass

    # ============================================================================
    # CONFIG
    # ============================================================================

    def _load_config(self):
        # First try to load from config.json
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
                livekit_url = config.get('livekit_url', '')
                api_key = config.get('api_key', '')
                api_secret = config.get('api_secret', '')
                room_name = config.get('room_name', 'voice-agent-test')
                llm_provider = config.get('llm_provider', 'openai/gpt-4o-mini')
                stt_provider = config.get('stt_provider', 'local/whisper-base')
                tts_provider = config.get('tts_provider', 'local/macos')

                # Load demo features
                if 'demo_features' in config:
                    demo_features.from_dict(config['demo_features'])
                    for attr, var in self.feature_vars.items():
                        var.set(getattr(demo_features, attr, False))
                    self.latency_slider.set(demo_features.latency_ms)

        except FileNotFoundError:
            # If config.json doesn't exist, load from .env file
            livekit_url = os.getenv('LIVEKIT_URL', '')
            api_key = os.getenv('LIVEKIT_API_KEY', '')
            api_secret = os.getenv('LIVEKIT_API_SECRET', '')
            room_name = 'voice-agent-test'
            llm_provider = os.getenv('DEFAULT_LLM', 'openai/gpt-4o-mini')
            stt_provider = os.getenv('DEFAULT_STT', 'local/whisper-base')
            tts_provider = os.getenv('DEFAULT_TTS', 'local/macos')

        # Insert values into GUI fields
        if livekit_url:
            self.livekit_url.insert(0, livekit_url)
        if api_key:
            self.api_key.insert(0, api_key)
        if api_secret:
            self.api_secret.insert(0, api_secret)
        self.room_name.insert(0, room_name)
        self.llm_provider.set(llm_provider)
        self.stt_provider.set(stt_provider)
        self.tts_provider.set(tts_provider)

    def _save_config(self):
        config = {
            'livekit_url': self.livekit_url.get(),
            'api_key': self.api_key.get(),
            'api_secret': self.api_secret.get(),
            'room_name': self.room_name.get(),
            'llm_provider': self.llm_provider.get(),
            'stt_provider': self.stt_provider.get(),
            'tts_provider': self.tts_provider.get(),
            'demo_features': demo_features.to_dict(),
        }
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)

    # ============================================================================
    # SESSION MANAGEMENT
    # ============================================================================

    def set_status(self, status, text, color):
        self.current_status = status
        self.root.after(0, lambda: self._draw_status_dot(color))
        self.root.after(0, lambda: self.status_text.config(text=text))

    def start_session(self):
        self._save_config()

        livekit_url = self.livekit_url.get()
        api_key = self.api_key.get()
        api_secret = self.api_secret.get()

        if not all([livekit_url, api_key, api_secret]):
            messagebox.showerror("Error", "Please fill in LiveKit URL, API Key, and API Secret")
            return

        self.is_running = True
        self.session_start_time = time.time()
        self.tool_calls = []
        self.user_messages = []
        self.agent_responses = []

        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.set_status('connecting', 'Connecting...', '#ffa500')

        self.log("System", "Starting voice session...", "system")

        # Reset timeline
        if demo_features.tool_timeline:
            self.timeline.reset()

        # Start audio meter
        if demo_features.audio_level_meter:
            self.audio_meter.start()

        # Run the agent
        async def run_agent():
            try:
                from livekit import rtc

                room = rtc.Room()
                await room.connect(
                    livekit_url,
                    self._create_token(api_key, api_secret, self.room_name.get())
                )

                self.set_status('listening', 'Listening', '#00ff88')
                self.log("System", "Connected. Listening...", "system")

                # Auto greeting
                if demo_features.auto_greeting:
                    self.log("System", "Sending auto-greeting...", "debug")

                while self.is_running:
                    await asyncio.sleep(0.1)

                # Summary
                if demo_features.conversation_summary:
                    self._show_summary()

                await room.disconnect()

            except Exception as e:
                self.log("System", f"Error: {str(e)}", "error")
                self.set_status('disconnected', 'Connection Failed', '#ff4757')
                self.start_btn.config(state='normal')
                self.stop_btn.config(state='disabled')
                self.audio_meter.stop()

        async_runner.run_coroutine(run_agent())

    def _create_token(self, api_key, api_secret, room_name):
        from livekit import api
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
        self.is_running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.set_status('disconnected', 'Disconnected', '#555555')
        self.audio_meter.stop()
        self.log("System", "Session ended.", "system")

    def speak_test(self, text):
        if self.is_running:
            self.user_messages.append({"text": text, "time": time.time()})
            self.stats_labels["user_msgs"].config(text=str(len(self.user_messages)))
            self.log("User", text, "user")
        else:
            self.log("System", "Start session first", "system")

    def get_session_stats(self):
        return {
            "duration_seconds": int(time.time() - self.session_start_time) if self.session_start_time else 0,
            "tool_calls": len(self.tool_calls),
            "user_messages": len(self.user_messages),
            "agent_responses": len(self.agent_responses),
            "avg_latency": sum(t["duration"] for t in self.tool_calls) / len(self.tool_calls) if self.tool_calls else 0,
        }

    def _show_summary(self):
        stats = self.get_session_stats()
        summary = f"""
╔═══════════════════════════════╗
║      SESSION SUMMARY          ║
╠═══════════════════════════════╣
║ Duration: {stats['duration_seconds']:>4}s              ║
║ Tool Calls: {stats['tool_calls']:>4}              ║
║ User Messages: {stats['user_messages']:>4}              ║
║ Agent Responses: {stats['agent_responses']:>4}              ║
║ Avg Latency: {stats['avg_latency']:>4.0f}ms           ║
╚═══════════════════════════════╝
"""
        self.log("Summary", summary.strip(), "system")


# ============================================================================
# CONSOLE MODE
# ============================================================================

async def console_entrypoint(ctx: JobContext) -> None:
    """Entrypoint for console mode testing."""
    await ctx.connect()

    global gui_instance
    gui_instance = ConsoleLogger()

    llm_provider = os.getenv("DEFAULT_LLM", "openai/gpt-4o-mini")
    stt_provider_str = os.getenv("DEFAULT_STT", "deepgram/nova-2")
    tts_provider_str = os.getenv("DEFAULT_TTS", "local/macos")

    print(f"[CONFIG] LLM: {llm_provider}, STT: {stt_provider_str}, TTS: {tts_provider_str}")

    # Get the appropriate STT and TTS provider instances
    stt_instance = get_stt_provider(stt_provider_str)
    print(f"[STT] Using: {type(stt_instance).__name__}")

    tts_instance = get_tts_provider(tts_provider_str)
    print(f"[TTS] Using: {type(tts_instance).__name__}")

    agent = VoiceAgentTest(llm_provider=llm_provider)

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=stt_instance,
        llm=llm_provider,
        tts=tts_instance,
    )

    print("[SESSION] Starting agent session...")

    await session.start(agent=agent, room=ctx.room)
    print("[SESSION] Ready to listen. Waiting for user input...")


class ConsoleLogger:
    """Simple console logger for tool calls."""
    def log_tool_execution(self, tool_name, args, result, duration_ms, is_error=False):
        tag = 'ERROR' if is_error else 'TOOL'
        print(f"[{tag}] {tool_name}({args}) -> {result} ({duration_ms:.0f}ms)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "console":
        global gui_instance
        gui_instance = ConsoleLogger()
        cli.run_app(WorkerOptions(entrypoint_fnc=console_entrypoint))
    else:
        root = tk.Tk()
        app = VoiceAgentGUI(root)

        def on_closing():
            async_runner.stop()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()


if __name__ == "__main__":
    main()
