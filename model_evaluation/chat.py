"""Terminal-based chat interface for MI-SLM Agent testing.

A Claude Code-inspired terminal UI with stunning visuals, streaming,
real-time trajectory visualization, and modern design for mechanistic
interpretability testing with Gemma Scope 2 SAE analysis.

Usage:
    uv run python -m model_evaluation.chat

Model Selection:
    On startup, choose from:
    1. Gemma 3 4B IT (bf16) - Full precision (VRAM calculated dynamically)
    2. Gemma 3 4B IT (int4) - 4-bit quantized (VRAM calculated dynamically)
    3. Gemma 3 12B IT (int4) - Larger model, 4-bit quantized (VRAM calculated dynamically)

Commands:
    /quit, /exit, /q - Exit the chat
    /clear - Clear conversation history
    /activations - Show SAE activation analysis
    /markdown - Switch to Markdown system prompt
    /plain - Switch to Plain Text system prompt
    /memory - Show GPU memory stats
    /stats - Show session statistics
    /help - Show available commands
    /qs, /questions - Browse test questions
    /ask <id> - Send a specific question (e.g., /ask p15, /ask s7m)
    /random - Send a random question
"""

import csv
import itertools
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

# Import config first - it loads .env at module import time, ensuring env vars
# (like HF_HUB_DISABLE_SYMLINKS_WARNING) are set before huggingface_hub is imported.
from model_evaluation.config import Settings
from model_evaluation.main_agent.gemma_model_loader import (
    GemmaModelConfig,
    MemoryTracker,
    get_gemma_model_id,
    load_gemma_model,
)
from model_evaluation.main_agent.gemma_scope_sae import load_gemma_scope_sae
from model_evaluation.main_agent.gemma_wrapper import GemmaWithSAE
from model_evaluation.main_agent.kb_generator import (
    GeneratorSession,
    create_kb_generator_agent,
)
from model_evaluation.main_agent.rag_agent import (
    MARKDOWN_SYSTEM_PROMPT,
    PLAIN_SYSTEM_PROMPT,
    create_safety_agent_with_tracing,
)
from model_evaluation.main_agent.tools import EvaluationContext

# ═══════════════════════════════════════════════════════════════════════════════
# Terminal & ANSI Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def clear_line() -> None:
    """Clear the current terminal line."""
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def hide_cursor() -> None:
    """Hide the terminal cursor."""
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()


def show_cursor() -> None:
    """Show the terminal cursor."""
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# Color System - Modern Gradient Palette
# ═══════════════════════════════════════════════════════════════════════════════


class Colors:
    """ANSI color codes with 256-color and RGB support."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"

    # Standard colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright variants
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Background colors
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"

    @staticmethod
    def rgb(r: int, g: int, b: int) -> str:
        """Create RGB foreground color."""
        return f"\033[38;2;{r};{g};{b}m"


# Brand colors for consistent styling
BRAND_PRIMARY = Colors.rgb(138, 180, 248)  # Soft blue
BRAND_SECONDARY = Colors.rgb(129, 212, 250)  # Cyan
BRAND_ACCENT = Colors.rgb(186, 104, 200)  # Purple
BRAND_SUCCESS = Colors.rgb(129, 199, 132)  # Green
BRAND_WARNING = Colors.rgb(255, 183, 77)  # Orange
BRAND_ERROR = Colors.rgb(239, 83, 80)  # Red
BRAND_INFO = Colors.rgb(100, 181, 246)  # Light blue
BRAND_MUTED = Colors.rgb(158, 158, 158)  # Gray


def styled(text: str, *styles: str) -> str:
    """Apply ANSI styles to text."""
    return "".join(styles) + text + Colors.RESET


def gradient_text(
    text: str,
    start_rgb: tuple[int, int, int],
    end_rgb: tuple[int, int, int],
) -> str:
    """Apply a gradient effect to text."""
    if not text:
        return text

    result = []
    length = len(text)

    for i, char in enumerate(text):
        if char == " ":
            result.append(char)
            continue

        ratio = i / max(length - 1, 1)
        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
        result.append(f"{Colors.rgb(r, g, b)}{char}")

    return "".join(result) + Colors.RESET


# ═══════════════════════════════════════════════════════════════════════════════
# Visual Components
# ═══════════════════════════════════════════════════════════════════════════════

# Progress bar characters
PROGRESS_FULL = "█"
PROGRESS_EMPTY = "░"

# Spinner frames
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# Icons
ICONS = {
    "check": "✓",
    "cross": "✗",
    "warning": "⚠",
    "lightning": "⚡",
    "brain": "🧠",
    "chart": "📊",
    "clock": "⏱",
    "sparkles": "✨",
    "robot": "🤖",
    "tools": "🔧",
    "cube": "📦",
    "terminal": "💻",
    "wave": "👋",
    "thought": "💭",
    "speech": "💬",
    "input": "📥",
    "output": "📤",
    "refresh": "🔄",
    "target": "🎯",
    "gpu": "🎮",
    "non-refusal": "🟢",
    "refusal": "🔴",
    "question": "❓",
    "list": "📋",
    "dice": "🎲",
    "select": "►",
    "model": "🤖",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Animated Spinner Class
# ═══════════════════════════════════════════════════════════════════════════════


class Spinner:
    """Animated spinner for async operations."""

    def __init__(
        self,
        *,
        message: str = "Loading",
        color: str = BRAND_PRIMARY,
    ) -> None:
        """Initialize spinner.

        Args:
            message: Text to display alongside spinner.
            color: ANSI color code for spinner.
        """
        self.message = message
        self.frames = SPINNER_FRAMES
        self.color = color
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.elapsed_time = 0.0

    def _animate(self) -> None:
        """Animation loop running in background thread."""
        start_time = time.time()
        for frame in itertools.cycle(self.frames):
            if self._stop_event.is_set():
                break
            self.elapsed_time = time.time() - start_time
            elapsed_str = f"{self.elapsed_time:.1f}s"
            spinner_char = styled(frame, self.color, Colors.BOLD)
            message_styled = styled(self.message, BRAND_MUTED)
            time_styled = styled(elapsed_str, Colors.DIM)
            sys.stdout.write(f"\r  {spinner_char} {message_styled} {time_styled}   ")
            sys.stdout.flush()
            time.sleep(0.08)

    def start(self) -> "Spinner":
        """Start the spinner animation."""
        hide_cursor()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def stop(self, *, success: bool = True, message: str | None = None) -> None:
        """Stop the spinner and show final status."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        show_cursor()
        clear_line()

        final_message = message or self.message
        elapsed_str = f"({self.elapsed_time:.1f}s)"

        if success:
            icon = styled(ICONS["check"], BRAND_SUCCESS, Colors.BOLD)
            msg = styled(final_message, BRAND_SUCCESS)
        else:
            icon = styled(ICONS["cross"], BRAND_ERROR, Colors.BOLD)
            msg = styled(final_message, BRAND_ERROR)

        time_styled = styled(elapsed_str, Colors.DIM)
        print(f"  {icon} {msg} {time_styled}")

    def __enter__(self) -> "Spinner":
        """Context manager entry."""
        return self.start()

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        exc_type = args[0] if args else None
        self.stop(success=exc_type is None)


# ═══════════════════════════════════════════════════════════════════════════════
# Visual Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════


def create_section_header(*, title: str, icon: str = "◆") -> str:
    """Create a section header with icon."""
    icon_styled = styled(icon, BRAND_ACCENT, Colors.BOLD)
    title_styled = styled(title, Colors.BOLD, Colors.WHITE)
    line = styled("─" * 50, Colors.DIM)
    return f"\n{icon_styled} {title_styled}\n{line}"


def create_stat_line(
    *,
    label: str,
    value: str,
    color: str = Colors.WHITE,
    icon: str | None = None,
) -> str:
    """Create a formatted statistics line."""
    bullet = styled("•", Colors.DIM)
    label_styled = styled(f"{label}:", Colors.DIM)
    value_styled = styled(value, color, Colors.BOLD)
    if icon:
        return f"  {icon} {label_styled} {value_styled}"
    return f"  {bullet} {label_styled} {value_styled}"


def create_progress_bar(
    *,
    value: float,
    max_value: float = 100,
    width: int = 40,
    show_percentage: bool = True,
    color_thresholds: dict[float, str] | None = None,
) -> str:
    """Create a progress bar with optional color coding."""
    percentage = (value / max_value * 100) if max_value > 0 else 0
    filled = int(width * percentage / 100)
    empty = width - filled

    # Determine color based on thresholds
    default_thresholds = {70: BRAND_SUCCESS, 85: BRAND_WARNING, 100: BRAND_ERROR}
    thresholds = color_thresholds or default_thresholds
    bar_color = BRAND_SUCCESS
    for threshold, color in sorted(thresholds.items()):
        if percentage <= threshold:
            bar_color = color
            break
        bar_color = color

    bar = PROGRESS_FULL * filled + PROGRESS_EMPTY * empty
    bar_styled = styled(bar, bar_color)
    brackets = styled("[", Colors.DIM), styled("]", Colors.DIM)

    if show_percentage:
        pct = styled(f"{percentage:5.1f}%", bar_color)
        return f"{brackets[0]}{bar_styled}{brackets[1]} {pct}"
    return f"{brackets[0]}{bar_styled}{brackets[1]}"


def create_mini_chart(*, values: list[float], width: int = 20) -> str:
    """Create a sparkline chart from values."""
    if not values:
        return styled("─" * width, Colors.DIM)

    blocks = " ▁▂▃▄▅▆▇█"
    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val or 1

    chart = ""
    for val in values[-width:]:
        normalized = (val - min_val) / val_range
        idx = int(normalized * (len(blocks) - 1))
        chart += blocks[idx]

    return styled(chart, BRAND_INFO)


def create_header(
    *,
    title: str,
    subtitle: str | None = None,
    width: int = 70,
) -> str:
    """Create a fancy header with gradient."""
    lines = []
    lines.append("")

    # ASCII art style header
    border = styled("═" * width, BRAND_PRIMARY)
    lines.append(border)

    # Title with gradient
    title_gradient = gradient_text(title.upper(), (138, 180, 248), (186, 104, 200))
    padding = (width - len(title)) // 2
    lines.append(" " * padding + title_gradient)

    if subtitle:
        sub_styled = styled(subtitle, Colors.DIM, Colors.ITALIC)
        sub_padding = (width - len(subtitle)) // 2
        lines.append(" " * sub_padding + sub_styled)

    lines.append(border)
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Session Statistics
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SessionStats:
    """Track statistics for the chat session."""

    total_turns: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tool_calls: int = 0
    total_latency_ms: float = 0.0
    prompt_style: str = "Markdown"
    system_prompt_tokens: int = 0
    agent_steps_history: list[int] = field(default_factory=list)
    latency_history: list[float] = field(default_factory=list)
    tokens_history: list[tuple[int, int]] = field(default_factory=list)  # (input, output)
    last_total_context_tokens: int = 0  # Full context (sys + tools + history + input)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per turn."""
        if self.total_turns > 0:
            return self.total_latency_ms / self.total_turns
        return 0.0

    @property
    def avg_steps(self) -> float:
        """Average agent steps per turn."""
        if not self.agent_steps_history:
            return 0.0
        return sum(self.agent_steps_history) / len(self.agent_steps_history)

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens generated per second."""
        if self.total_latency_ms == 0:
            return 0.0
        return self.total_output_tokens / (self.total_latency_ms / 1000)


# ═══════════════════════════════════════════════════════════════════════════════
# Display Functions
# ═══════════════════════════════════════════════════════════════════════════════


def get_memory_stats() -> dict[str, float]:
    """Get current GPU memory statistics."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "peak": 0, "free": 0, "total": 0}

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - allocated

    return {
        "allocated": allocated,
        "reserved": reserved,
        "peak": peak,
        "free": free,
        "total": total,
    }


def display_memory_stats() -> None:
    """Display GPU memory statistics with visual flair."""
    stats = get_memory_stats()

    print(create_section_header(title="GPU Memory", icon=ICONS["gpu"]))

    if stats["total"] == 0:
        print(styled("  No GPU available", BRAND_WARNING))
        return

    print(
        create_stat_line(
            label="Allocated",
            value=f"{stats['allocated']:.2f} GB",
            color=BRAND_SUCCESS,
        )
    )
    print(
        create_stat_line(
            label="Reserved",
            value=f"{stats['reserved']:.2f} GB",
            color=BRAND_WARNING,
        )
    )
    print(
        create_stat_line(
            label="Peak",
            value=f"{stats['peak']:.2f} GB",
            color=BRAND_ACCENT,
        )
    )
    print(
        create_stat_line(
            label="Free",
            value=f"{stats['free']:.2f} GB",
            color=BRAND_INFO,
        )
    )
    print(
        create_stat_line(
            label="Total",
            value=f"{stats['total']:.2f} GB",
            color=Colors.WHITE,
        )
    )

    # Visual memory bar
    usage_pct = (stats["allocated"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    bar = create_progress_bar(value=usage_pct, width=45)
    print(f"\n  {bar}")


def display_session_stats(stats: SessionStats) -> None:
    """Display comprehensive session statistics."""
    print(create_section_header(title="Session Statistics", icon=ICONS["chart"]))

    # Prompt info
    print(
        create_stat_line(
            label="Prompt Style",
            value=stats.prompt_style,
            color=BRAND_PRIMARY,
        )
    )
    print(
        create_stat_line(
            label="System Tokens",
            value=f"{stats.system_prompt_tokens:,}",
            color=Colors.WHITE,
        )
    )

    # Conversation stats
    print(
        create_stat_line(
            label="Total Turns",
            value=str(stats.total_turns),
            color=BRAND_SUCCESS,
            icon=ICONS["speech"],
        )
    )
    print(
        create_stat_line(
            label="Input Tokens",
            value=f"{stats.total_input_tokens:,}",
            color=BRAND_WARNING,
            icon=ICONS["input"],
        )
    )
    print(
        create_stat_line(
            label="Output Tokens",
            value=f"{stats.total_output_tokens:,}",
            color=BRAND_ACCENT,
            icon=ICONS["output"],
        )
    )
    print(
        create_stat_line(
            label="Context Size",
            value=f"{stats.last_total_context_tokens:,}",
            color=BRAND_ERROR,  # Detailed context usage
            icon=ICONS["list"],
        )
    )
    print(
        create_stat_line(
            label="Tool Calls",
            value=str(stats.total_tool_calls),
            color=BRAND_INFO,
            icon=ICONS["tools"],
        )
    )

    # Performance stats
    print(
        create_stat_line(
            label="Avg Latency",
            value=f"{stats.avg_latency_ms:.0f} ms",
            color=Colors.WHITE,
            icon=ICONS["clock"],
        )
    )
    print(
        create_stat_line(
            label="Tokens/sec",
            value=f"{stats.tokens_per_second:.1f}",
            color=BRAND_SUCCESS,
            icon=ICONS["lightning"],
        )
    )
    print(
        create_stat_line(
            label="Avg Steps",
            value=f"{stats.avg_steps:.1f}",
            color=BRAND_PRIMARY,
            icon=ICONS["refresh"],
        )
    )

    # Latency sparkline if we have history
    if stats.latency_history:
        chart = create_mini_chart(values=stats.latency_history, width=30)
        print(f"\n  {styled('Latency trend:', Colors.DIM)} {chart}")


def display_activation_summary(model: GemmaWithSAE) -> None:
    """Display detailed SAE activation analysis with visual enhancements."""
    activations = model.last_activations

    if activations is None:
        warn_icon = styled(ICONS["warning"], BRAND_WARNING)
        warn_msg = styled("No activations recorded yet.", BRAND_WARNING)
        print(f"\n  {warn_icon} {warn_msg}")
        return

    print(create_section_header(title="SAE Activation Analysis", icon=ICONS["brain"]))

    # Token breakdown
    answer_tokens = len(activations.tokens) - activations.prompt_len
    print(
        create_stat_line(
            label="Total Tokens",
            value=str(len(activations.tokens)),
            color=Colors.WHITE,
        )
    )
    print(
        create_stat_line(
            label="Prompt Tokens",
            value=str(activations.prompt_len),
            color=BRAND_WARNING,
        )
    )
    print(
        create_stat_line(
            label="Answer Tokens",
            value=str(answer_tokens),
            color=BRAND_SUCCESS,
        )
    )

    # SAE Metrics with visual indicators
    print(create_section_header(title="SAE Metrics", icon="📐"))

    # L0 metric with color coding
    if activations.l0 < 80:
        l0_color = BRAND_SUCCESS
    elif activations.l0 < 120:
        l0_color = BRAND_WARNING
    else:
        l0_color = BRAND_ERROR

    l0_thresholds = {40: BRAND_SUCCESS, 60: BRAND_INFO, 80: BRAND_WARNING, 100: BRAND_ERROR}
    l0_bar = create_progress_bar(
        value=activations.l0,
        max_value=200,
        width=25,
        show_percentage=False,
        color_thresholds=l0_thresholds,
    )
    l0_label = styled("L0:", Colors.DIM)
    l0_val = styled(f"{activations.l0:.1f}", l0_color, Colors.BOLD)
    print(f"  {l0_label} {l0_val} features/token {l0_bar}")

    # FVU metric
    if activations.fvu < 0.05:
        fvu_color = BRAND_SUCCESS
    elif activations.fvu < 0.1:
        fvu_color = BRAND_WARNING
    else:
        fvu_color = BRAND_ERROR

    fvu_thresholds = {5: BRAND_SUCCESS, 10: BRAND_WARNING, 20: BRAND_ERROR}
    fvu_bar = create_progress_bar(
        value=activations.fvu * 100,
        max_value=20,
        width=25,
        show_percentage=False,
        color_thresholds=fvu_thresholds,
    )
    fvu_label = styled("FVU:", Colors.DIM)
    fvu_val = styled(f"{activations.fvu:.2%}", fvu_color, Colors.BOLD)
    print(f"  {fvu_label} {fvu_val} reconstruction {fvu_bar}")

    # Top features at decision point
    print(create_section_header(title="Top Features @ Decision Point", icon=ICONS["target"]))

    last_idx = len(activations.tokens) - 1
    top_feats = activations.top_features[last_idx].tolist()[:10]
    top_acts = activations.top_activations[last_idx].tolist()[:10]

    max_act = max(top_acts) if top_acts else 1.0

    # Create a nice table-like display
    rank_colors = [BRAND_ACCENT, BRAND_PRIMARY, BRAND_INFO]
    for i, (feat, act) in enumerate(zip(top_feats, top_acts, strict=True)):
        rank_color = rank_colors[i] if i < 3 else Colors.WHITE
        rank = styled(f"#{i + 1:<2}", rank_color, Colors.BOLD)

        feat_str = styled(f"{feat:>6}", BRAND_ACCENT)
        act_normalized = act / max_act
        bar_width = 20
        filled = int(bar_width * act_normalized)
        bar = PROGRESS_FULL * filled + PROGRESS_EMPTY * (bar_width - filled)
        bar_color = BRAND_SUCCESS if act_normalized > 0.5 else BRAND_INFO
        bar_styled = styled(bar, bar_color)

        act_str = styled(f"{act:.4f}", Colors.WHITE)
        print(f"  {rank} Feature {feat_str} {bar_styled} {act_str}")


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming Trajectory Visualization
# ═══════════════════════════════════════════════════════════════════════════════


class StreamingTrajectory:
    """Manages real-time visualization of agent trajectory during streaming."""

    def __init__(self) -> None:
        """Initialize trajectory tracker."""
        self.current_node: str | None = None
        self.tool_calls_made: list[str] = []
        self.agent_steps = 0
        self.streamed_tokens = 0
        self.start_time = time.time()
        self._response_started = False
        self.final_response: str | None = None
        self.chunks_retrieved: list[dict] = []
        self._pending_step_display: bool = False
        self._last_node_was_tools: bool = False

    def display_node_change(self, *, node_name: str) -> None:
        """Display when agent transitions to a new node.

        For the "model" node, we defer showing "Model thinking..." until we see
        a tool call. If no tool call is made (i.e., final response generation),
        we skip the step display entirely since the response streams immediately.
        """
        if node_name == self.current_node:
            return

        previous_node = self.current_node
        self.current_node = node_name

        if node_name == "model":
            self.agent_steps += 1
            # Defer step display - will show if tool calls are made
            self._pending_step_display = True
            self._last_node_was_tools = previous_node == "tools"
        elif node_name == "tools":
            tools_icon = styled(ICONS["tools"], BRAND_INFO)
            exec_msg = styled("Executing tools...", BRAND_INFO)
            print(f"\n  {tools_icon} {exec_msg}")

    def _show_pending_step(self) -> None:
        """Show the pending step display if not yet shown."""
        if self._pending_step_display:
            self._pending_step_display = False
            step_badge = styled(
                f" Step {self.agent_steps} ",
                Colors.BG_BLUE,
                Colors.WHITE,
            )
            reasoning_icon = styled(ICONS["thought"], BRAND_ACCENT)
            print(f"\n  {step_badge} {reasoning_icon} Reasoning...")

    def display_tool_call(self, *, tool_name: str, tool_args: dict) -> None:
        """Display a tool call being made."""
        # Show pending step display now that we know there's a tool call
        self._show_pending_step()

        self.tool_calls_made.append(tool_name)
        tool_styled = styled(tool_name, BRAND_PRIMARY, Colors.BOLD)
        args_str = ", ".join(f"{k}={v!r}" for k, v in tool_args.items())
        args_styled = styled(f"({args_str})", Colors.DIM)
        arrow = styled("→", BRAND_INFO)
        print(f"    {arrow} {tool_styled}{args_styled}")

    def display_tool_result(self, *, tool_name: str, result: str) -> None:
        """Display tool execution result."""
        check = styled(ICONS["check"], BRAND_SUCCESS)
        name_styled = styled(tool_name, BRAND_SUCCESS)
        # Truncate long results
        result_display = result[:100] + "..." if len(result) > 100 else result
        result_styled = styled(result_display, Colors.DIM)
        print(f"    {check} {name_styled}: {result_styled}")

    def display_search_results(self, *, event_data: dict) -> None:
        """Display search results metadata from knowledge base search.

        Args:
            event_data: Custom stream event data from search_knowledge_base tool.
        """
        event_type = event_data.get("event", "")

        if event_type == "search_start":
            query = event_data.get("query", "")
            num_results = event_data.get("num_results", 5)
            search_icon = styled("🔍", BRAND_INFO)
            query_styled = styled(f'"{query}"', BRAND_PRIMARY)
            print(f"    {search_icon} Searching for {query_styled} (top {num_results})")

        elif event_type == "search_complete":
            chunks_found = event_data.get("chunks_found", 0)
            chunks = event_data.get("chunks", [])

            if chunks_found == 0:
                warn_icon = styled(ICONS["warning"], BRAND_WARNING)
                print(f"    {warn_icon} No chunks found")
                return

            # Store chunks for later reference
            self.chunks_retrieved = chunks

            # Display chunk summary
            docs_icon = styled("📚", BRAND_SUCCESS)
            count_styled = styled(str(chunks_found), BRAND_SUCCESS, Colors.BOLD)
            print(f"    {docs_icon} Found {count_styled} relevant chunks:")
            print("")

            # Display each chunk's metadata
            for chunk in chunks:
                rank = chunk.get("rank", "?")
                score = chunk.get("score", 0.0)
                title = chunk.get("document_title", "Unknown")
                section = chunk.get("section", "")
                subsection = chunk.get("subsection", "")
                privacy = chunk.get("privacy_level", "public")

                # Privacy badge with color
                if privacy == "private":
                    privacy_badge = styled(f"[{privacy.upper()}]", BRAND_ERROR, Colors.BOLD)
                elif privacy == "mixed":
                    privacy_badge = styled(f"[{privacy.upper()}]", BRAND_WARNING, Colors.BOLD)
                else:
                    privacy_badge = styled(f"[{privacy.upper()}]", BRAND_SUCCESS)

                # Score color based on relevance
                if score < 0.5:
                    score_color = BRAND_SUCCESS
                elif score < 1.0:
                    score_color = BRAND_WARNING
                else:
                    score_color = BRAND_ERROR
                score_styled = styled(f"{score:.3f}", score_color)

                # Build source path
                source_parts = [title]
                if section:
                    source_parts.append(section)
                if subsection:
                    source_parts.append(subsection)
                source_path = " › ".join(source_parts)

                # Truncate source path if too long
                max_source_len = 50
                if len(source_path) > max_source_len:
                    source_path = source_path[: max_source_len - 3] + "..."

                source_styled = styled(source_path, Colors.DIM)

                # Rank badge
                rank_badge = styled(f"#{rank}", BRAND_ACCENT)

                print(f"      {rank_badge} {privacy_badge} score={score_styled} {source_styled}")

            print("")

    def start_response_stream(self) -> None:
        """Initialize response streaming display."""
        if not self._response_started:
            self._response_started = True
            robot = styled(ICONS["robot"], BRAND_SUCCESS)
            agent_label = styled("Agent", BRAND_SUCCESS, Colors.BOLD)
            print(f"\n  {robot} {agent_label}:")
            print("")
            sys.stdout.write("    ")

    def stream_token(self, *, token: str) -> None:
        """Stream a single token to output."""
        self.start_response_stream()
        self.streamed_tokens += 1

        # Handle newlines with proper indentation
        if "\n" in token:
            token = token.replace("\n", "\n    ")

        sys.stdout.write(token)
        sys.stdout.flush()

    def finish_response(self) -> None:
        """Finish the streaming response display."""
        if self._response_started:
            print()  # Final newline

    def display_final_response(self, *, response: str) -> None:
        """Display the final response if streaming didn't show it."""
        if self._response_started:
            # Already streamed, don't duplicate
            return

        self.final_response = response

        if not response or not response.strip():
            warn_icon = styled(ICONS["warning"], BRAND_WARNING)
            warn_msg = styled("No response generated.", BRAND_WARNING)
            print(f"\n  {warn_icon} {warn_msg}")
            return

        # Display response with nice formatting
        robot = styled(ICONS["robot"], BRAND_SUCCESS)
        agent_label = styled("Agent", BRAND_SUCCESS, Colors.BOLD)
        print(f"\n  {robot} {agent_label}:")
        print("")

        # Indent response text
        for line in response.split("\n"):
            print(f"    {line}")

    def display_summary(
        self,
        *,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Display turn summary after completion."""
        border_color = BRAND_MUTED

        # Determine latency color
        if latency_ms < 2000:
            latency_color = BRAND_SUCCESS
        elif latency_ms < 5000:
            latency_color = BRAND_WARNING
        else:
            latency_color = BRAND_ERROR

        print("")
        print(f"  {styled('╭' + '─' * 55, border_color)}")

        # Latency
        clock = ICONS["clock"]
        latency_val = styled(f"{latency_ms:,.0f}ms", latency_color, Colors.BOLD)
        tps = output_tokens / (latency_ms / 1000) if latency_ms > 0 else 0
        tps_styled = styled(f"({tps:.1f} tok/s)", Colors.DIM)
        print(f"  {styled('│', border_color)} {clock} {latency_val} {tps_styled}")

        # Tokens
        in_icon = ICONS["input"]
        out_icon = ICONS["output"]
        in_tokens_styled = styled(f"{input_tokens:,}", BRAND_WARNING)
        out_tokens_styled = styled(f"{output_tokens:,}", BRAND_SUCCESS)
        border = styled("│", border_color)
        print(f"  {border} {in_icon} {in_tokens_styled} → {out_icon} {out_tokens_styled} tokens")

        # Steps
        steps_icon = ICONS["refresh"]
        steps_val = styled(str(self.agent_steps), BRAND_ACCENT, Colors.BOLD)
        print(f"  {styled('│', border_color)} {steps_icon} {steps_val} agent steps")

        # Tool calls
        if self.tool_calls_made:
            tools_icon = ICONS["tools"]
            tools_str = styled(", ".join(self.tool_calls_made), BRAND_INFO)
            print(f"  {styled('│', border_color)} {tools_icon} {tools_str}")

        print(f"  {styled('╰' + '─' * 55, border_color)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Welcome Banner & Help
# ═══════════════════════════════════════════════════════════════════════════════


def display_welcome_banner() -> None:
    """Display the welcome banner with ASCII art."""
    # MI-SLM ASCII art banner
    banner_lines = [
        "    ███╗   ███╗██╗      ███████╗██╗     ███╗   ███╗",
        "    ████╗ ████║██║      ██╔════╝██║     ████╗ ████║",
        "    ██╔████╔██║██║█████╗███████╗██║     ██╔████╔██║",
        "    ██║╚██╔╝██║██║╚════╝╚════██║██║     ██║╚██╔╝██║",
        "    ██║ ╚═╝ ██║██║      ███████║███████╗██║ ╚═╝ ██║",
        "    ╚═╝     ╚═╝╚═╝      ╚══════╝╚══════╝╚═╝     ╚═╝",
    ]

    # Print with gradient effect line by line
    lines = banner_lines
    start_color = (138, 180, 248)  # Blue
    end_color = (186, 104, 200)  # Purple

    for i, line in enumerate(lines):
        ratio = i / max(len(lines) - 1, 1)
        r = int(start_color[0] + (end_color[0] - start_color[0]) * ratio)
        g = int(start_color[1] + (end_color[1] - start_color[1]) * ratio)
        b = int(start_color[2] + (end_color[2] - start_color[2]) * ratio)
        print(styled(line, Colors.rgb(r, g, b)))
        time.sleep(0.02)  # Subtle animation delay

    # Subtitles - centered
    subtitle1 = "Small Language Model Agent Testing Interface"
    subtitle2 = "feat. Gemma Scope 2 SAE Analysis"
    print("")
    # Center based on banner width (~52 chars)
    padding1 = (52 - len(subtitle1)) // 2
    padding2 = (52 - len(subtitle2)) // 2
    print(styled(" " * padding1 + subtitle1, Colors.DIM, Colors.ITALIC))
    print(styled(" " * padding2 + subtitle2, Colors.DIM))
    print("")


def display_commands_help() -> None:
    """Display available commands in a styled format."""
    commands = [
        ("/quit", "Exit the chat", "q"),
        ("/clear", "Clear conversation history", None),
        ("/activations", "Show SAE activation analysis", None),
        ("/markdown", "Switch to Markdown prompt", None),
        ("/plain", "Switch to Plain Text prompt", None),
        ("/private", "KB generates private docs (default)", None),
        ("/public", "KB generates public docs only", None),
        ("/memory", "Show GPU memory stats", None),
        ("/stats", "Show session statistics", None),
        ("/help", "Show this help message", None),
    ]

    question_commands = [
        ("/qs", "Browse test questions", "questions"),
        ("/qs refusal", "Filter to refusal questions", None),
        ("/qs non-refusal", "Filter to non-refusal questions", None),
        ("/ask <id>", "Send question (e.g., /ask s7m, /ask s1)", None),
        ("/random", "Send a random question", None),
        ("/random refusal", "Send a random refusal question", None),
    ]

    print(create_section_header(title="Available Commands", icon=ICONS["terminal"]))

    for cmd, desc, alias in commands:
        cmd_styled = styled(cmd, BRAND_PRIMARY, Colors.BOLD)
        desc_styled = styled(desc, Colors.WHITE)
        if alias:
            alias_styled = styled(f"(/{alias})", Colors.DIM)
            print(f"  {cmd_styled:<25} {desc_styled} {alias_styled}")
        else:
            print(f"  {cmd_styled:<25} {desc_styled}")

    print("")
    print(create_section_header(title="Question Testing", icon=ICONS["question"]))

    for cmd, desc, alias in question_commands:
        cmd_styled = styled(cmd, BRAND_ACCENT, Colors.BOLD)
        desc_styled = styled(desc, Colors.WHITE)
        if alias:
            alias_styled = styled(f"(/{alias})", Colors.DIM)
            print(f"  {cmd_styled:<25} {desc_styled} {alias_styled}")
        else:
            print(f"  {cmd_styled:<25} {desc_styled}")

    print("")


# ═══════════════════════════════════════════════════════════════════════════════
# Question Bank for Testing
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Question:
    """A single test question from the question bank."""

    number: int
    text: str
    is_malicious: bool
    category: str  # "papers" or "synthetic"
    origin: str | None = None  # For papers questions (legacy)
    universe_context: str | None = None  # For synthetic questions (new format)

    @property
    def id(self) -> str:
        """Generate unique ID like 'p15' or 's7m'."""
        prefix = "p" if self.category == "papers" else "s"
        suffix = "m" if self.is_malicious else ""
        return f"{prefix}{self.number}{suffix}"


class QuestionBank:
    """Loads and manages test questions from CSV files."""

    def __init__(self, *, questions_dir: Path) -> None:
        """Initialize the question bank.

        Args:
            questions_dir: Path to the directory containing question CSV files.
        """
        self.questions: list[Question] = []
        self._load_questions(questions_dir=questions_dir)

    def _load_questions(self, *, questions_dir: Path) -> None:
        """Load questions from CSV files.

        Supports both legacy format (Document of origin, Malicious question)
        and new format (Universe Context, Is Refusal).

        Args:
            questions_dir: Path to the directory containing question CSV files.
        """
        csv_files = [
            ("papers_questions.csv", "papers"),
            ("synthetic_questions.csv", "synthetic"),
        ]

        for filename, category in csv_files:
            filepath = questions_dir / filename
            if not filepath.exists():
                continue

            with open(filepath, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Detect format based on available columns
                    if "Universe Context" in row:
                        # New format (synthetic questions)
                        question = Question(
                            number=int(row["Number"]),
                            text=row["Question"].strip(),
                            is_malicious=row["Is Refusal"].strip().lower() == "yes",
                            category=category,
                            universe_context=row["Universe Context"].strip(),
                        )
                    else:
                        # Legacy format (papers questions)
                        question = Question(
                            number=int(row["Number"]),
                            text=row["Question"].strip(),
                            origin=row["Document of origin"].strip(),
                            is_malicious=row["Malicious question"].strip().lower() == "yes",
                            category=category,
                        )
                    self.questions.append(question)

    def filter(
        self,
        *,
        category: str | None = None,
        malicious: bool | None = None,
    ) -> list[Question]:
        """Filter questions by criteria.

        Args:
            category: Filter by category ("papers" or "synthetic"), or None for all.
            malicious: Filter by malicious status, or None for all.

        Returns:
            List of questions matching the criteria.
        """
        result = self.questions

        if category is not None:
            result = [q for q in result if q.category == category]

        if malicious is not None:
            result = [q for q in result if q.is_malicious == malicious]

        return result

    def get_by_id(self, *, question_id: str) -> Question | None:
        """Get a question by its unique ID.

        Args:
            question_id: Question ID like 'p15', 's7m', 'p1m', etc.

        Returns:
            The matching question, or None if not found.
        """
        # Parse ID format: prefix (p/s) + number + optional 'm' suffix
        match = re.match(r"^([ps])(\d+)(m?)$", question_id.lower())
        if not match:
            return None

        prefix, number_str, suffix = match.groups()
        category = "papers" if prefix == "p" else "synthetic"
        number = int(number_str)
        is_malicious = suffix == "m"

        for q in self.questions:
            if q.category == category and q.number == number and q.is_malicious == is_malicious:
                return q

        return None

    def random(
        self,
        *,
        category: str | None = None,
        malicious: bool | None = None,
    ) -> Question | None:
        """Get a random question matching criteria.

        Args:
            category: Filter by category, or None for all.
            malicious: Filter by malicious status, or None for all.

        Returns:
            A random question matching criteria, or None if no matches.
        """
        filtered = self.filter(category=category, malicious=malicious)
        if not filtered:
            return None
        return random.choice(filtered)  # noqa: S311


@dataclass
class QuestionBrowserState:
    """State for the question browser pagination and filters."""

    page: int = 1
    page_size: int = 10
    category: str | None = None
    malicious: bool | None = None


def display_questions_browser(
    *,
    questions: list[Question],
    state: QuestionBrowserState,
) -> None:
    """Display a paginated question browser.

    Args:
        questions: List of questions to display.
        state: Current browser state with pagination and filters.
    """
    total = len(questions)
    total_pages = max(1, (total + state.page_size - 1) // state.page_size)
    state.page = max(1, min(state.page, total_pages))

    start_idx = (state.page - 1) * state.page_size
    end_idx = min(start_idx + state.page_size, total)
    page_questions = questions[start_idx:end_idx]

    # Build filter description
    filters = []
    if state.category:
        filters.append(state.category)
    else:
        filters.append("all")
    if state.malicious is True:
        filters.append("refusal")
    elif state.malicious is False:
        filters.append("non-refusal")
    filter_text = " │ ".join(filters)

    # Header
    print("")
    header = gradient_text(
        f"{ICONS['list']} Questions Browser",
        (138, 180, 248),
        (186, 104, 200),
    )
    print(f"  {header}")
    print(f"  {styled('─' * 100, Colors.DIM)}")

    # Table Header
    header = f" {styled('ID', Colors.BOLD):<6} │ {styled('Question', Colors.BOLD):<68} │ {styled('Universe Context', Colors.BOLD)}"
    print(header)
    print(f"  {styled('─' * 100, Colors.DIM)}")

    # Filter info
    filter_label = styled("Showing:", Colors.DIM)
    filter_value = styled(filter_text, BRAND_INFO)
    page_info = styled(f"Page {state.page}/{total_pages}", BRAND_MUTED)
    total_info = styled(f"({total} questions)", Colors.DIM)
    print(f"  {filter_label} {filter_value} │ {page_info} {total_info}")
    print("")

    if not page_questions:
        print(f"  {styled('No questions match the current filters.', Colors.DIM)}")
        print("")
        return

    # Question list
    for q in page_questions:
        # Determine types for display
        # Icon based on refusal status
        icon = ICONS["refusal"] if q.is_malicious else ICONS["non-refusal"]

        # Text style based on refusal status
        id_style = BRAND_ERROR if q.is_malicious else BRAND_SUCCESS

        # ID Column
        id_text = f"{icon} {q.id:<4}"

        # Question Column (truncate if too long)
        q_text = (q.text[:65] + "...") if len(q.text) > 65 else q.text

        # Universe Context Column
        # Use universe_context if available, otherwise fallback to legacy origin or "Unknown"
        context_text = q.universe_context or q.origin or "Unknown"
        context_text = (context_text[:20] + "...") if len(context_text) > 20 else context_text

        print(f" {styled(id_text, id_style)} │ {q_text:<68} │ {styled(context_text, Colors.DIM)}")

    print("")

    # Navigation hints
    nav_hints = []
    if state.page < total_pages:
        nav_hints.append(styled("/qs next", BRAND_PRIMARY))
    if state.page > 1:
        nav_hints.append(styled("/qs prev", BRAND_PRIMARY))
    nav_hints.append(styled("/qs all", BRAND_PRIMARY))
    nav_hints.append(styled("/ask <id>", BRAND_ACCENT))

    hints_text = " │ ".join(nav_hints)
    print(f"  {styled('Commands:', Colors.DIM)} {hints_text}")
    print("")


def _display_width(text: str) -> int:
    """Calculate the display width of text, accounting for wide characters like emojis.

    Emojis and other wide characters typically display as 2 columns in terminals,
    but Python's len() counts them as 1 or 2 code points depending on the emoji.

    Args:
        text: Text to measure.

    Returns:
        Estimated display width in terminal columns.
    """
    width = 0
    for char in text:
        code_point = ord(char)
        # Emoji ranges (simplified - covers most common emojis)
        if (
            0x1F300 <= code_point <= 0x1F9FF  # Misc Symbols, Emoticons, etc.
            or 0x2600 <= code_point <= 0x26FF  # Misc Symbols
            or 0x2700 <= code_point <= 0x27BF  # Dingbats
            or 0x1F600 <= code_point <= 0x1F64F  # Emoticons
            or 0x1F680 <= code_point <= 0x1F6FF  # Transport symbols
        ):
            width += 2  # Emojis are typically 2 columns wide
        else:
            width += 1
    return width


def _wrap_text(text: str, *, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width.

    Args:
        text: Text to wrap.
        max_width: Maximum width per line.

    Returns:
        List of wrapped lines.
    """
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        # If word itself is too long, split it
        if len(word) > max_width:
            if current_line:
                lines.append(current_line)
                current_line = ""
            # Split long word
            while len(word) > max_width:
                lines.append(word[:max_width])
                word = word[max_width:]
            if word:
                current_line = word
        elif len(current_line) + len(word) + 1 <= max_width:
            current_line = f"{current_line} {word}".strip()
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


def _print_box_line(
    *,
    text: str,
    border_color: str,
    content_width: int = 58,
    text_color: str | None = None,
    display_width: int | None = None,
) -> None:
    """Print a single line inside the box with proper padding.

    Args:
        text: Text content to print.
        border_color: Color for the border.
        content_width: Width available for content.
        text_color: Optional color for the text.
        display_width: Pre-calculated display width (for emoji-aware centering).
    """
    # Use provided display width or calculate from text
    actual_width = display_width if display_width is not None else _display_width(text)
    padding = content_width - actual_width
    left_border = styled("│", border_color)
    right_border = styled("│", border_color)
    content = styled(text, text_color) if text_color else text
    print(f"  {left_border}  {content}{' ' * max(0, padding)}{right_border}")


def _print_wrapped_field(
    *,
    label: str,
    value: str,
    border_color: str,
    value_color: str,
    content_width: int = 58,
    label_width: int = 10,
) -> None:
    """Print a labeled field with word-wrapping for long values.

    Args:
        label: Field label (e.g., "Origin:").
        value: Field value to display.
        border_color: Color for the border.
        value_color: Color for the value text.
        content_width: Total width available for content.
        label_width: Width reserved for the label column.
    """
    value_width = content_width - label_width - 2  # -2 for spacing

    # Wrap the value text
    wrapped_lines = _wrap_text(value, max_width=value_width)

    if not wrapped_lines:
        wrapped_lines = [""]

    # First line includes the label
    first_line_value = wrapped_lines[0] if wrapped_lines else ""
    label_styled = styled(label, Colors.DIM)
    value_styled = styled(first_line_value, value_color)

    # Calculate padding for first line
    first_line_raw = f"{label:<{label_width}} {first_line_value}"
    padding = content_width - len(first_line_raw)

    left_border = styled("│", border_color)
    right_border = styled("│", border_color)
    print(
        f"  {left_border}  {label_styled:<{label_width + len(label_styled) - len(label)}} "
        f"{value_styled}{' ' * max(0, padding)}{right_border}"
    )

    # Continuation lines (indented to align with value)
    for line in wrapped_lines[1:]:
        line_styled = styled(line, value_color)
        indent = " " * (label_width + 1)
        line_raw = f"{indent}{line}"
        padding = content_width - len(line_raw)
        print(f"  {left_border}  {indent}{line_styled}{' ' * max(0, padding)}{right_border}")


def display_question_detail(*, question: Question) -> None:
    """Display full details of a question before sending.

    Args:
        question: The question to display.
    """
    print("")

    box_width = 60
    content_width = box_width - 2  # Account for    # Determine styles based on refusal status
    border_color = BRAND_ERROR if question.is_malicious else BRAND_SUCCESS
    icon = ICONS["refusal"] if question.is_malicious else ICONS["non-refusal"]

    # Top border
    print(f"  {styled('╭' + '─' * box_width + '╮', border_color)}")

    # Title (centered, accounting for emoji width)
    title_text = f"{icon} Question {question.id}"
    # Use display width for centering (emojis are 2 columns wide)
    title_display_width = _display_width(title_text)
    title_padding = (content_width - title_display_width) // 2
    centered_title = " " * title_padding + title_text
    # Pass the display width for correct right-side padding
    _print_box_line(
        text=centered_title,
        border_color=border_color,
        content_width=content_width,
        display_width=title_padding + title_display_width,
    )

    # Separator
    print(f"  {styled('├' + '─' * box_width + '┤', border_color)}")

    # Metadata fields with word-wrapping
    _print_wrapped_field(
        label="Category:",
        value=question.category.capitalize(),
        border_color=border_color,
        value_color=BRAND_INFO,
        content_width=content_width,
    )

    # Show origin or universe context depending on question type
    source_label = "Universe:" if question.universe_context else "Origin:"
    source_value = question.universe_context or question.origin or "Unknown"
    _print_wrapped_field(
        label=source_label,
        value=source_value,
        border_color=border_color,
        value_color=BRAND_INFO,
        content_width=content_width,
    )

    type_value = (
        "Refusal (attempts private access)"
        if question.is_malicious
        else "Non-Refusal (legitimate question)"
    )
    type_color = BRAND_ERROR if question.is_malicious else BRAND_SUCCESS
    _print_wrapped_field(
        label="Type:",
        value=type_value,
        border_color=border_color,
        value_color=type_color,
        content_width=content_width,
    )

    # Separator
    print(f"  {styled('├' + '─' * box_width + '┤', border_color)}")

    # Question text (word-wrapped)
    question_lines = _wrap_text(question.text, max_width=content_width - 2)

    for line in question_lines:
        _print_box_line(text=line, border_color=border_color, text_color=Colors.WHITE)

    # Bottom border
    print(f"  {styled('╰' + '─' * box_width + '╯', border_color)}")

    # Status message
    print("")
    sending_msg = gradient_text(
        f"{ICONS['lightning']} Sending question...",
        (129, 199, 132),
        (129, 212, 250),
    )
    print(f"  {sending_msg}")
    print("")


# ═══════════════════════════════════════════════════════════════════════════════
# Model Selection Menu
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class VRAMBreakdown:
    """Detailed VRAM breakdown by component."""

    model_params_gb: float
    kv_cache_gb: float
    activations_gb: float
    sae_gb: float
    overhead_gb: float
    total_gb: float


@dataclass
class ModelOption:
    """A model configuration option for the selection menu."""

    key: str
    name: str
    size: str
    quantization: str | None
    description: str
    vram_estimate: str
    vram_breakdown: VRAMBreakdown


# ═══════════════════════════════════════════════════════════════════════════════
# VRAM Estimation (based on https://apxml.com/posts/how-to-calculate-vram-requirements-for-an-llm)
# ═══════════════════════════════════════════════════════════════════════════════

# Model parameter counts (in billions)
MODEL_PARAMS: dict[str, float] = {
    "1b": 1.0,
    "4b": 4.0,
    "12b": 12.0,
    "27b": 27.0,
}

# Bytes per parameter for each precision
BYTES_PER_PARAM: dict[str | None, float] = {
    None: 2.0,  # bf16 (default, no quantization)
    "bf16": 2.0,
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}

# Fixed VRAM components (in GB)
SAE_VRAM_GB: float = 0.5  # SAE with 16k width
OVERHEAD_FACTOR: float = 0.15  # 15% overhead for framework, CUDA kernels, etc.


def estimate_vram_breakdown(
    *,
    model_size: str,
    quantization: str | None = None,
    context_length: int = 8192,
    batch_size: int = 1,
) -> VRAMBreakdown:
    """Estimate VRAM requirements for inference with detailed breakdown.

    Formula based on: https://apxml.com/posts/how-to-calculate-vram-requirements-for-an-llm
    Total VRAM ≈ VRAM_params + VRAM_kv_cache + VRAM_activations + VRAM_sae + VRAM_overhead

    Args:
        model_size: Model size identifier (e.g., "4b", "12b").
        quantization: Quantization type ("int4", "int8", None for bf16).
        context_length: Maximum context length for KV cache estimation.
        batch_size: Batch size for inference.

    Returns:
        VRAMBreakdown with all components.
    """
    # Get parameter count
    params_billions = MODEL_PARAMS.get(model_size, 4.0)
    params = params_billions * 1e9

    # Get bytes per parameter
    bytes_per_param = BYTES_PER_PARAM.get(quantization, 2.0)

    # Model parameters VRAM
    vram_params_gb = (params * bytes_per_param) / (1024**3)

    # KV Cache estimation (simplified)
    # For Gemma models: ~32 layers, 8 KV heads, 128 head dim for 4B
    # KV cache ≈ 2 × layers × kv_heads × head_dim × seq_len × batch × bytes
    # Using approximate values scaled by model size
    layers = int(16 * params_billions / 4)  # Scale layers with model size
    kv_heads = 8
    head_dim = 128
    kv_bytes = 2  # KV cache typically in fp16/bf16

    vram_kv_cache = (2 * layers * kv_heads * head_dim * context_length * batch_size * kv_bytes) / (
        1024**3
    )

    # Activations estimation (rough approximation)
    # Activations scale with model size and context
    hidden_dim = int(2048 * (params_billions / 4) ** 0.5)  # Approximate hidden dim
    vram_activations_gb = (batch_size * context_length * hidden_dim * layers * 2 * 0.001) / (
        1024**3
    )  # Simplified heuristic

    # SAE VRAM (fixed for 16k width)
    vram_sae_gb = SAE_VRAM_GB

    # Sum base VRAM
    base_vram = vram_params_gb + vram_kv_cache + vram_activations_gb + vram_sae_gb

    # Calculate overhead
    overhead_gb = base_vram * OVERHEAD_FACTOR

    # Total VRAM
    total_vram = base_vram + overhead_gb

    return VRAMBreakdown(
        model_params_gb=vram_params_gb,
        kv_cache_gb=vram_kv_cache,
        activations_gb=vram_activations_gb,
        sae_gb=vram_sae_gb,
        overhead_gb=overhead_gb,
        total_gb=total_vram,
    )


def estimate_vram_gb(
    *,
    model_size: str,
    quantization: str | None = None,
    context_length: int = 8192,
    batch_size: int = 1,
) -> float:
    """Estimate total VRAM requirements for inference.

    Args:
        model_size: Model size identifier (e.g., "4b", "12b").
        quantization: Quantization type ("int4", "int8", None for bf16).
        context_length: Maximum context length for KV cache estimation.
        batch_size: Batch size for inference.

    Returns:
        Estimated total VRAM in gigabytes.
    """
    breakdown = estimate_vram_breakdown(
        model_size=model_size,
        quantization=quantization,
        context_length=context_length,
        batch_size=batch_size,
    )
    return breakdown.total_gb


def format_vram_estimate(vram_gb: float) -> str:
    """Format VRAM estimate as a human-readable string.

    Args:
        vram_gb: VRAM in gigabytes.

    Returns:
        Formatted string like "~5.2 GB".
    """
    return f"~{vram_gb:.1f} GB"


def create_model_options() -> list[ModelOption]:
    """Create model options with computed VRAM estimates.

    Returns:
        List of ModelOption with dynamically calculated VRAM estimates.
    """
    breakdown_1b = estimate_vram_breakdown(model_size="1b")
    breakdown_4b = estimate_vram_breakdown(model_size="4b")
    breakdown_12b = estimate_vram_breakdown(model_size="12b")

    return [
        ModelOption(
            key="1",
            name="Gemma 3 1B IT (bf16)",
            size="1b",
            quantization=None,
            description="Ultra-lightweight, low memory",
            vram_estimate=format_vram_estimate(breakdown_1b.total_gb),
            vram_breakdown=breakdown_1b,
        ),
        ModelOption(
            key="2",
            name="Gemma 3 4B IT (bf16)",
            size="4b",
            quantization=None,
            description="Balanced speed/quality",
            vram_estimate=format_vram_estimate(breakdown_4b.total_gb),
            vram_breakdown=breakdown_4b,
        ),
        ModelOption(
            key="3",
            name="Gemma 3 12B IT (bf16)",
            size="12b",
            quantization=None,
            description="High quality, more VRAM",
            vram_estimate=format_vram_estimate(breakdown_12b.total_gb),
            vram_breakdown=breakdown_12b,
        ),
    ]


# Initialize model options with computed VRAM estimates
MODEL_OPTIONS: list[ModelOption] = create_model_options()


def display_model_selection_menu() -> ModelOption:
    """Display model selection menu and return chosen option.

    Returns:
        The selected ModelOption.
    """
    print("")
    # Centered header
    menu_width = 62
    header_text = f"{ICONS['model']} Select Model Configuration"
    header = gradient_text(
        header_text,
        (138, 180, 248),
        (186, 104, 200),
    )
    # Center the header (account for icon taking ~2 chars)
    header_padding = (menu_width - len(header_text)) // 2
    print(f"{' ' * header_padding}{header}")
    print(f"  {styled('─' * (menu_width - 2), Colors.DIM)}")
    print("")

    for option in MODEL_OPTIONS:
        # Key badge
        key_badge = styled(f" {option.key} ", Colors.BOLD, Colors.BG_BLUE, Colors.WHITE)

        # Model name
        name_styled = styled(option.name, BRAND_PRIMARY, Colors.BOLD)

        # VRAM estimate
        vram_styled = styled(f"[{option.vram_estimate}]", BRAND_WARNING)

        # Description
        desc_styled = styled(option.description, Colors.DIM)

        print(f"  {key_badge}  {name_styled:<30} {vram_styled}")
        print(f"       {desc_styled}")

        # VRAM breakdown display
        b = option.vram_breakdown
        tree_branch = styled("├─", Colors.DIM)
        tree_end = styled("└─", Colors.DIM)
        sep = styled("│", Colors.DIM)

        # Line 1: Model params and KV cache
        params_label = styled("Model:", Colors.DIM)
        params_val = styled(f"{b.model_params_gb:.1f} GB", BRAND_INFO)
        kv_label = styled("KV cache:", Colors.DIM)
        kv_val = styled(f"{b.kv_cache_gb:.1f} GB", BRAND_INFO)
        print(f"       {tree_branch} {params_label} {params_val}  {sep}  {kv_label} {kv_val}")

        # Line 2: Activations, SAE, and overhead
        act_label = styled("Activations:", Colors.DIM)
        act_val = styled(f"{b.activations_gb:.1f} GB", BRAND_INFO)
        sae_label = styled("SAE:", Colors.DIM)
        sae_val = styled(f"{b.sae_gb:.1f} GB", BRAND_INFO)
        overhead_pct = int(OVERHEAD_FACTOR * 100)
        overhead_text = styled(f"+{overhead_pct}% overhead", Colors.DIM)
        print(
            f"       {tree_end} {act_label} {act_val}  {sep}  {sae_label} {sae_val}  {sep}  "
            f"{overhead_text}"
        )
        print("")

    print(f"  {styled('─' * (menu_width - 2), Colors.DIM)}")

    # Prompt for selection
    prompt_icon = styled(ICONS["select"], BRAND_ACCENT)
    prompt_text = styled("Enter choice (1-3):", Colors.WHITE)

    while True:
        try:
            choice = input(f"  {prompt_icon} {prompt_text} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("")
            raise

        # Find matching option
        for option in MODEL_OPTIONS:
            if choice == option.key:
                check = styled(ICONS["check"], BRAND_SUCCESS)
                selected_msg = styled(f"Selected: {option.name}", BRAND_SUCCESS)
                print(f"\n  {check} {selected_msg}")
                print("")
                return option

        # Invalid choice
        cross = styled(ICONS["cross"], BRAND_ERROR)
        err_msg = styled("Invalid choice. Please enter 1, 2, or 3.", BRAND_ERROR)
        print(f"  {cross} {err_msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_model(settings: Settings) -> GemmaWithSAE:
    """Load the GemmaWithSAE model with animated progress."""
    print(create_header(title="Loading Model Stack", subtitle="Initializing Gemma + SAE"))

    # Get the correct model ID
    model_id = get_gemma_model_id(
        size=settings.gemma_model_size,
        model_type=settings.gemma_model_type,
    )

    # Load base model
    with Spinner(message="Loading Gemma model") as spinner:
        with MemoryTracker("Loading Gemma model"):
            config = GemmaModelConfig(
                model_id=model_id,
                size=settings.gemma_model_size,
                max_context_length=settings.gemma_max_context_length,
                tokenizer_id=settings.gemma_tokenizer_id,
            )
            model, tokenizer = load_gemma_model(config, token=settings.hf_token)
            model.eval()
        spinner.stop(success=True, message="Gemma model loaded")

    device = str(next(model.parameters()).device)
    print(create_stat_line(label="Device", value=device, color=BRAND_INFO, icon=ICONS["gpu"]))
    print(
        create_stat_line(
            label="Model",
            value=model_id,
            color=BRAND_PRIMARY,
            icon=ICONS["cube"],
        )
    )

    # Load SAE
    with Spinner(message="Loading SAE", color=BRAND_ACCENT) as spinner:
        with MemoryTracker("Loading SAE"):
            sae, sae_config = load_gemma_scope_sae(
                model_size=settings.gemma_model_size,
                model_type=settings.gemma_model_type,
                layer=settings.effective_sae_layer,
                width=settings.sae_width,
                l0_size=settings.sae_l0_size,
                device=device,
            )
        spinner.stop(success=True, message="SAE loaded")

    print(
        create_stat_line(
            label="SAE Layer",
            value=str(settings.effective_sae_layer),
            color=BRAND_ACCENT,
            icon=ICONS["brain"],
        )
    )
    print(create_stat_line(label="SAE Width", value=settings.sae_width, color=BRAND_INFO))

    # Create wrapper
    gemma_with_sae = GemmaWithSAE(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        sae_config=sae_config,
        max_tokens=settings.max_new_tokens,
    )

    # Success message
    print("")
    success_msg = gradient_text(
        f"{ICONS['sparkles']} Model stack ready!",
        (129, 199, 132),
        (129, 212, 250),
    )
    print(f"  {success_msg}")
    print("")

    display_memory_stats()

    return gemma_with_sae


def count_tokens(model: GemmaWithSAE, text: str) -> int:
    """Count tokens in text using the model's tokenizer."""
    return len(model._tokenizer.encode(text))


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming Chat Loop
# ═══════════════════════════════════════════════════════════════════════════════


def run_chat(model: GemmaWithSAE, *, use_markdown: bool = True) -> None:
    """Run the interactive chat loop with streaming support."""
    agent = create_safety_agent_with_tracing(model, use_markdown_rules=use_markdown)

    # Initialize KB Generator session for the search_knowledge_base tool
    print(f"\n  {styled('Initializing KB Generator Agent...', Colors.DIM)}")
    generator_agent, generator_checkpointer = create_kb_generator_agent(enable_tracing=True)
    generator_session = GeneratorSession(
        agent=generator_agent,
        checkpointer=generator_checkpointer,
    )
    include_private_info = True  # Default to private mode for safety testing

    # Initialize stats
    stats = SessionStats(prompt_style="Markdown" if use_markdown else "Plain Text")
    system_prompt = MARKDOWN_SYSTEM_PROMPT if use_markdown else PLAIN_SYSTEM_PROMPT
    stats.system_prompt_tokens = count_tokens(model, system_prompt)

    # Initialize question bank
    questions_dir = Path(__file__).parent / "questions"
    question_bank = QuestionBank(questions_dir=questions_dir)
    browser_state = QuestionBrowserState()

    # Display mode header
    mode_badge = styled(f" {stats.prompt_style} ", Colors.BOLD, Colors.BG_BLUE, Colors.WHITE)
    print(f"\n  {styled('Active Mode:', Colors.DIM)} {mode_badge}")
    sys_tokens_msg = f"System prompt: {stats.system_prompt_tokens:,} tokens"
    print(f"  {styled(sys_tokens_msg, Colors.DIM)}")

    # Display KB generator privacy mode
    privacy_badge = (
        styled(" Private ", BRAND_ERROR)
        if include_private_info
        else styled(" Public ", BRAND_SUCCESS)
    )
    print(f"  {styled('KB Privacy Mode:', Colors.DIM)} {privacy_badge}")

    # Display question bank info
    total_qs = len(question_bank.questions)
    refusal_qs = len(question_bank.filter(malicious=True))
    non_refusal_qs = len(question_bank.filter(malicious=False))
    qs_info = (
        f"Question bank: {total_qs} questions ({non_refusal_qs} non-refusal, {refusal_qs} refusal)"
    )

    print(f"  {styled(qs_info, Colors.DIM)}")

    display_commands_help()

    conversation_state: dict[str, list[BaseMessage]] = {"messages": []}

    # Prompt styling
    prompt_icon = styled("›", BRAND_PRIMARY, Colors.BOLD)
    user_label = styled("You", BRAND_PRIMARY, Colors.BOLD)

    # Track current question for context-aware document generation
    current_question: Question | None = None

    while True:
        try:
            print("")
            user_input = input(f"  {prompt_icon} {user_label}: ").strip()
        except (KeyboardInterrupt, EOFError):
            wave = styled(ICONS["wave"], BRAND_PRIMARY)
            goodbye = styled("Goodbye!", BRAND_PRIMARY, Colors.BOLD)
            print(f"\n\n  {wave} {goodbye}")
            break

        if not user_input:
            continue

        # Handle commands
        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "/q"):
            wave = styled(ICONS["wave"], BRAND_PRIMARY)
            goodbye = styled("Goodbye!", BRAND_PRIMARY, Colors.BOLD)
            print(f"\n  {wave} {goodbye}")
            break

        if cmd == "/clear":
            conversation_state = {"messages": []}
            check = styled(ICONS["check"], BRAND_SUCCESS)
            msg = styled("Conversation cleared.", BRAND_SUCCESS)
            print(f"\n  {check} {msg}")
            continue

        if cmd == "/activations":
            display_activation_summary(model)
            continue

        if cmd == "/memory":
            display_memory_stats()
            continue

        if cmd == "/stats":
            display_session_stats(stats)
            continue

        if cmd == "/help":
            display_commands_help()
            continue

        if cmd == "/markdown":
            agent = create_safety_agent_with_tracing(model, use_markdown_rules=True)
            stats.prompt_style = "Markdown"
            stats.system_prompt_tokens = count_tokens(model, MARKDOWN_SYSTEM_PROMPT)
            mode_badge = styled(" Markdown ", Colors.BOLD, Colors.BG_BLUE, Colors.WHITE)
            check = styled(ICONS["check"], BRAND_SUCCESS)
            print(f"\n  {check} Switched to {mode_badge}")
            continue

        if cmd == "/plain":
            agent = create_safety_agent_with_tracing(model, use_markdown_rules=False)
            stats.prompt_style = "Plain Text"
            stats.system_prompt_tokens = count_tokens(model, PLAIN_SYSTEM_PROMPT)
            mode_badge = styled(" Plain Text ", Colors.BOLD, Colors.BG_MAGENTA, Colors.WHITE)
            check = styled(ICONS["check"], BRAND_SUCCESS)
            print(f"\n  {check} Switched to {mode_badge}")
            continue

        if cmd == "/private":
            include_private_info = True
            mode_badge = styled(" Private ", Colors.BOLD, BRAND_ERROR)
            check = styled(ICONS["check"], BRAND_SUCCESS)
            print(f"\n  {check} KB Generator will include {mode_badge} documents")
            continue

        if cmd == "/public":
            include_private_info = False
            mode_badge = styled(" Public ", Colors.BOLD, BRAND_SUCCESS)
            check = styled(ICONS["check"], BRAND_SUCCESS)
            print(f"\n  {check} KB Generator will only include {mode_badge} documents")
            continue

        # Question browser commands
        if cmd in ("/qs", "/questions"):
            filtered = question_bank.filter(
                category=browser_state.category,
                malicious=browser_state.malicious,
            )
            display_questions_browser(questions=filtered, state=browser_state)
            continue

        if cmd.startswith("/qs ") or cmd.startswith("/questions "):
            parts = cmd.split()
            subcommand = parts[1] if len(parts) > 1 else ""

            if subcommand == "next":
                browser_state.page += 1
            elif subcommand == "prev":
                browser_state.page = max(1, browser_state.page - 1)
            elif subcommand == "papers":
                browser_state.category = "papers"
                browser_state.page = 1
            elif subcommand == "synthetic":
                browser_state.category = "synthetic"
                browser_state.page = 1
            elif subcommand == "refusal":
                browser_state.malicious = True
                browser_state.page = 1
            elif subcommand == "non-refusal":
                browser_state.malicious = False
                browser_state.page = 1
            elif subcommand == "all":
                browser_state.category = None
                browser_state.malicious = None
                browser_state.page = 1
            else:
                cross = styled(ICONS["cross"], BRAND_ERROR)
                err_msg = styled(f"Unknown filter: {subcommand}", BRAND_ERROR)
                print(f"\n  {cross} {err_msg}")
                continue

            filtered = question_bank.filter(
                category=browser_state.category,
                malicious=browser_state.malicious,
            )
            display_questions_browser(questions=filtered, state=browser_state)
            continue

        # Ask specific question
        if cmd.startswith("/ask "):
            parts = cmd.split()
            if len(parts) < 2:
                cross = styled(ICONS["cross"], BRAND_ERROR)
                err_msg = styled("Usage: /ask <id> (e.g., /ask p15, /ask s7m)", BRAND_ERROR)
                print(f"\n  {cross} {err_msg}")
                continue

            question_id = parts[1]
            question = question_bank.get_by_id(question_id=question_id)
            if question is None:
                cross = styled(ICONS["cross"], BRAND_ERROR)
                err_msg = styled(f"Question '{question_id}' not found", BRAND_ERROR)
                print(f"\n  {cross} {err_msg}")
                continue

            display_question_detail(question=question)
            user_input = question.text
            current_question = question  # Set for context-aware doc generation
            # Fall through to normal message processing

        # Random question
        if cmd.startswith("/random"):
            parts = cmd.split()
            malicious_filter: bool | None = None
            category_filter: str | None = None

            for part in parts[1:]:
                if part == "refusal":
                    malicious_filter = True
                elif part == "non-refusal":
                    malicious_filter = False
                elif part == "papers":
                    category_filter = "papers"
                elif part == "synthetic":
                    category_filter = "synthetic"

            question = question_bank.random(
                category=category_filter,
                malicious=malicious_filter,
            )
            if question is None:
                cross = styled(ICONS["cross"], BRAND_ERROR)
                err_msg = styled("No questions match the criteria", BRAND_ERROR)
                print(f"\n  {cross} {err_msg}")
                continue

            display_question_detail(question=question)
            user_input = question.text
            current_question = question  # Set for context-aware doc generation
            # Fall through to normal message processing

        # Input token counting is now handled internally by GemmaWithSAE

        # Add user message to conversation
        conversation_state["messages"].append(HumanMessage(content=user_input))

        # Stream agent response with trajectory visualization
        try:
            trajectory = StreamingTrajectory()
            start_time = time.perf_counter()
            # Collect all new messages from agent updates during streaming
            new_messages: list[BaseMessage] = []

            # Use streaming mode for real-time output (including custom events)
            # Pass universe context if we have a question with one
            universe_ctx = current_question.universe_context if current_question else None
            eval_context = EvaluationContext(
                include_private_info=include_private_info,
                generator_session=generator_session,
                universe_context=universe_ctx,
            )
            for stream_mode, chunk in agent.stream(
                conversation_state,
                stream_mode=["messages", "updates", "custom"],
                context=eval_context,
            ):
                if stream_mode == "messages":
                    token, metadata = chunk
                    node = metadata.get("langgraph_node", "")

                    # Track node changes
                    trajectory.display_node_change(node_name=node)

                    # Stream AI message tokens (works with API models)
                    if isinstance(token, AIMessageChunk) and token.content:
                        trajectory.stream_token(token=str(token.content))
                        # output_tokens tracked by model wrapper

                elif stream_mode == "custom":
                    # Handle custom stream events (e.g., from search_knowledge_base)
                    if isinstance(chunk, dict) and "event" in chunk:
                        trajectory.display_search_results(event_data=chunk)

                elif stream_mode == "updates":
                    for source, update in chunk.items():
                        # Collect messages from each update for conversation history
                        update_messages = update.get("messages", [])

                        # Handle completed messages
                        if source == "model":
                            for msg in update_messages:
                                if isinstance(msg, AIMessage):
                                    new_messages.append(msg)
                                    # Capture final response content
                                    if msg.content and not msg.tool_calls:
                                        trajectory.final_response = str(msg.content)
                                        # output_tokens tracked by model wrapper
                                    # Display tool calls
                                    if msg.tool_calls:
                                        for tc in msg.tool_calls:
                                            trajectory.display_tool_call(
                                                tool_name=tc["name"],
                                                tool_args=tc.get("args", {}),
                                            )

                        elif source == "tools":
                            for msg in update_messages:
                                if isinstance(msg, ToolMessage):
                                    new_messages.append(msg)
                                    content = str(msg.content) if msg.content else ""
                                    trajectory.display_tool_result(
                                        tool_name=msg.name or "tool",
                                        result=content,
                                    )

            # Update token stats from model wrapper (source of truth)
            stats.last_total_context_tokens = model.last_input_token_count
            stats.total_input_tokens = model.total_input_tokens
            stats.total_output_tokens = model.total_output_tokens

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Finish streaming display and show final response
            trajectory.finish_response()

            # Display final response (handles local models that don't stream tokens)
            if trajectory.final_response:
                trajectory.display_final_response(response=trajectory.final_response)

            # Update session stats
            stats.total_turns += 1
            # Note: total_input/output_tokens are already synced from model wrapper above

            # Calculate turn tokens for history/summary
            # Use context size as "input cost" for this turn
            turn_input_tokens = model.last_input_token_count

            # Calculate output tokens for this turn (approximate from final response)
            turn_output_tokens = 0
            if trajectory.final_response:
                turn_output_tokens = count_tokens(model, trajectory.final_response)

            stats.total_latency_ms += latency_ms
            stats.agent_steps_history.append(trajectory.agent_steps)
            stats.latency_history.append(latency_ms)
            stats.tokens_history.append((turn_input_tokens, turn_output_tokens))

            # Display summary
            trajectory.display_summary(
                latency_ms=latency_ms,
                input_tokens=turn_input_tokens,
                output_tokens=turn_output_tokens,
            )

            # Update conversation state by appending all new messages from this turn
            # We collected AIMessage and ToolMessage instances during streaming
            conversation_state["messages"].extend(new_messages)

            # Reset current question after processing
            current_question = None

        except Exception as e:
            cross = styled(ICONS["cross"], BRAND_ERROR)
            err_msg = styled(f"Error: {e}", BRAND_ERROR)
            print(f"\n  {cross} {err_msg}")


def main() -> None:
    """Main entry point with startup animation."""
    try:
        # Clear screen and show banner
        print("\033[2J\033[H", end="")  # Clear screen
        display_welcome_banner()

        # Show model selection menu
        selected_model = display_model_selection_menu()

        # Load settings and override with selected model configuration
        settings = Settings()  # type: ignore[call-arg]
        settings.gemma_model_size = selected_model.size  # type: ignore[assignment]
        settings.gemma_quantization = selected_model.quantization  # type: ignore[assignment]

        # Load model with selected configuration
        model = load_model(settings)

        # Start chat
        run_chat(model, use_markdown=True)

    except KeyboardInterrupt:
        wave = styled(ICONS["wave"], BRAND_PRIMARY)
        msg = styled("Interrupted. Goodbye!", BRAND_PRIMARY)
        print(f"\n\n  {wave} {msg}")
        sys.exit(0)
    except Exception as e:
        cross = styled(ICONS["cross"], BRAND_ERROR)
        err_msg = styled(f"Failed to start: {e}", BRAND_ERROR)
        print(f"\n  {cross} {err_msg}")
        sys.exit(1)
    finally:
        show_cursor()


if __name__ == "__main__":
    main()
