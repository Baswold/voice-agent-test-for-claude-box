#!/usr/bin/env python3
"""
Voice Agent Test Application
A simple voice agent to test clean tool execution with varying complexity levels.
"""

import asyncio
import json
import os
import random
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
# QUICK TOOLS (< 1 second)
# ============================================================================

@function_tool
async def get_current_time(context: RunContext[AgentSession], timezone: str = "UTC") -> str:
    """Returns the current time in the user's timezone. Use this when the user asks what time it is."""
    start_time = time.time()
    result = datetime.now().astimezone().strftime("%I:%M %p")
    elapsed = (time.time() - start_time) * 1000
    print(f"[TOOL] get_current_time({timezone}) -> {result} ({elapsed:.0f}ms)")
    return f"The current time is {result}."


@function_tool
async def flip_coin(context: RunContext[AgentSession]) -> str:
    """Flips a coin and returns heads or tails. Use this when the user wants to flip a coin."""
    start_time = time.time()
    result = random.choice(["heads", "tails"])
    elapsed = (time.time() - start_time) * 1000
    print(f"[TOOL] flip_coin() -> {result} ({elapsed:.0f}ms)")
    return f"It's {result}."


@function_tool
async def roll_dice(context: RunContext[AgentSession], sides: int = 6) -> str:
    """Rolls a die with the specified number of sides (default 6). Use this when the user wants to roll dice."""
    start_time = time.time()
    if sides < 2 or sides > 100:
        sides = 6
    result = random.randint(1, sides)
    elapsed = (time.time() - start_time) * 1000
    print(f"[TOOL] roll_dice({sides}) -> {result} ({elapsed:.0f}ms)")
    return f"You rolled a {result}."


# ============================================================================
# MEDIUM TOOLS (1-3 seconds)
# ============================================================================

@function_tool
async def get_weather(context: RunContext[AgentSession], location: str) -> str:
    """Gets the current weather for a location. Use this when the user asks about the weather."""
    start_time = time.time()
    # Simulate API delay
    await asyncio.sleep(1.5)

    # Mock weather data
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "windy"]
    temp = random.randint(45, 95)
    condition = random.choice(conditions)

    elapsed = (time.time() - start_time) * 1000
    print(f"[TOOL] get_weather('{location}') -> {temp}°F {condition} ({elapsed:.0f}ms)")
    return f"It's currently {temp} degrees Fahrenheit and {condition} in {location}."


@function_tool
async def calculator(context: RunContext[AgentSession], expression: str) -> str:
    """Evaluates a mathematical expression. Use this when the user needs to do math calculations.
    The expression should be a string like '2 + 2' or '234 * 567 + 89'."""
    start_time = time.time()
    try:
        # Safe evaluation of mathematical expressions
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names)
        elapsed = (time.time() - start_time) * 1000
        print(f"[TOOL] calculator('{expression}') -> {result} ({elapsed:.0f}ms)")
        return f"The result is {result}."
    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        print(f"[TOOL] calculator('{expression}') -> ERROR: {e} ({elapsed:.0f}ms)")
        return f"Sorry, I couldn't calculate that. The error was: {str(e)}"


@function_tool
async def web_search(context: RunContext[AgentSession], query: str) -> str:
    """Searches the web for information. Use this when the user asks to search for something online."""
    start_time = time.time()
    # Simulate search delay
    await asyncio.sleep(2.0)

    # Mock search results
    results = [
        f"Here's what I found about '{query}':",
        f"1. Wikipedia: {query} is a topic of interest with many applications.",
        f"2. News: Recent updates about {query} have been trending.",
        f"3. Overview: {query} relates to various fields including technology and science.",
    ]

    elapsed = (time.time() - start_time) * 1000
    print(f"[TOOL] web_search('{query}') -> {len(results)} results ({elapsed:.0f}ms)")
    return "\n".join(results)


# ============================================================================
# LONG/COMPLEX TOOLS (3-10 seconds)
# ============================================================================

@function_tool
async def analyze_text(context: RunContext[AgentSession], text: str, analysis_type: str = "sentiment") -> str:
    """Analyzes text using AI. The analysis_type can be 'sentiment', 'summary', 'keywords', or 'translation'."""
    start_time = time.time()
    # Simulate LLM processing delay
    await asyncio.sleep(4.0)

    text_preview = text[:50] + "..." if len(text) > 50 else text

    if analysis_type == "sentiment":
        sentiments = ["positive", "negative", "neutral"]
        result = f"Sentiment analysis of '{text_preview}': {random.choice(sentiments)} (confidence: {random.randint(70, 99)}%)"
    elif analysis_type == "summary":
        result = f"Summary of '{text_preview}': The text discusses key topics and presents various viewpoints on the subject matter."
    elif analysis_type == "keywords":
        keywords = ["technology", "innovation", "analysis", "development", "research", "trends"]
        result = f"Keywords from '{text_preview}': {', '.join(random.sample(keywords, 3))}"
    elif analysis_type == "translation":
        result = f"Translation of '{text_preview}': [Translated to Spanish/French/German based on content]"
    else:
        result = f"Analysis complete for '{text_preview}' with type: {analysis_type}"

    elapsed = (time.time() - start_time) * 1000
    print(f"[TOOL] analyze_text('{text_preview}', '{analysis_type}') -> completed ({elapsed:.0f}ms)")
    return result


@function_tool
async def generate_report(context: RunContext[AgentSession], topic: str, sections: list[str], detail_level: str = "brief") -> str:
    """Generates a detailed report on a topic. Provide sections as a list like ['intro', 'analysis', 'conclusion'].
    Detail level can be 'brief' or 'detailed'."""
    start_time = time.time()
    # Simulate report generation delay
    await asyncio.sleep(6.0)

    section_names = ", ".join(sections)
    result = f"""Report on {topic}:

Sections: {section_names}
Detail Level: {detail_level}

{'. '.join([f"{s.capitalize()}: Analysis and findings for this section demonstrate key insights related to {topic}."
           for s in sections[:3]])}

Overall conclusion: The report highlights important aspects of {topic} with actionable recommendations."""

    elapsed = (time.time() - start_time) * 1000
    print(f"[TOOL] generate_report('{topic}', {sections}, '{detail_level}') -> completed ({elapsed:.0f}ms)")
    return result


@function_tool
async def database_query(context: RunContext[AgentSession], sql: str, filters: dict[str, Any] | None = None) -> str:
    """Executes a database query with optional filters. SQL should be a SELECT statement.
    Filters is a dictionary like {'age': '>25', 'status': 'active'}."""
    start_time = time.time()
    # Simulate database operation delay
    await asyncio.sleep(5.0)

    filters_str = json.dumps(filters) if filters else "none"
    # Mock query results
    result_count = random.randint(5, 50)
    result = f"""Query executed successfully:

SQL: {sql}
Filters: {filters_str}

Results: {result_count} records found

Sample results:
- Record 1: Matches your criteria
- Record 2: Meets all filter conditions
- Record 3: Within specified parameters"""

    elapsed = (time.time() - start_time) * 1000
    print(f"[TOOL] database_query('{sql}', {filters_str}) -> {result_count} records ({elapsed:.0f}ms)")
    return result


# ============================================================================
# EDGE CASE TOOL (Error Testing)
# ============================================================================

@function_tool
async def trigger_error(context: RunContext[AgentSession], error_type: str = "api_error") -> str:
    """Triggers different types of errors for testing error handling.
    Error types: 'timeout', 'api_error', 'invalid_input', 'network_error'."""
    start_time = time.time()

    error_messages = {
        "timeout": "Request timed out. The service took too long to respond.",
        "api_error": "API error: Invalid credentials or rate limit exceeded.",
        "invalid_input": "Invalid input: The provided parameters are not valid.",
        "network_error": "Network error: Unable to reach the server.",
    }

    elapsed = (time.time() - start_time) * 1000
    print(f"[TOOL] trigger_error('{error_type}') -> simulating error ({elapsed:.0f}ms)")
    return f"I encountered an issue: {error_messages.get(error_type, 'Unknown error occurred')}. Please try again."


# ============================================================================
# VOICE AGENT
# ============================================================================

class VoiceAgentTest(Agent):
    """A voice agent for testing clean tool execution."""

    def __init__(self, llm_provider: str = "openai/gpt-4o-mini"):
        # Build full instructions with silent tool execution directive
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

    async def on_enter(self) -> None:
        """Called when the agent starts."""
        print("[AGENT] Agent started. Greeting user...")
        # Don't generate a greeting here - let the user start the conversation


# ============================================================================
# SERVER AND ENTRYPOINT
# ============================================================================

async def entrypoint(ctx: JobContext) -> None:
    """Main entrypoint for the voice agent."""
    await ctx.connect()

    # Get configuration from environment
    llm_provider = os.getenv("DEFAULT_LLM", "openai/gpt-4o-mini")
    stt_provider = os.getenv("DEFAULT_STT", "deepgram/nova-2")
    tts_provider = os.getenv("DEFAULT_TTS", "cartesia/sonic-english")

    print(f"[CONFIG] LLM: {llm_provider}, STT: {stt_provider}, TTS: {tts_provider}")

    # Create the agent with all tools
    agent = VoiceAgentTest(llm_provider=llm_provider)

    # Create session with VAD and plugins
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=stt_provider,
        llm=llm_provider,
        tts=tts_provider,
    )

    print("[SESSION] Starting agent session...")

    # Start the session
    await session.start(agent=agent, room=ctx.room)

    # Start listening immediately without greeting
    print("[SESSION] Ready to listen. Waiting for user input...")


if __name__ == "__main__":
    # For development: Run with console mode for local testing
    # python voice_agent.py console
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
