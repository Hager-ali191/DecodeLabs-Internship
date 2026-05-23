"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          DECODELABS AI INTERNSHIP — PROJECT 1: RULE-BASED AI CHATBOT        ║
║                    The Logic Engine | Batch 2026                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture: IPO Model (Input → Process → Output)
  Phase 1 — INPUT:    Raw user text → Sanitized (lowercased + stripped)
  Phase 2 — PROCESS:  O(1) Dictionary lookup (Hash Map, not If-Elif Ladder)
  Phase 3 — OUTPUT:   Matched response or graceful fallback

Key Design Decisions (per training material):
  ✅ while True infinite loop — the "heartbeat" of the system
  ✅ .lower().strip() sanitization — handles HeLLo, hello, HELLO uniformly
  ✅ dict.get() with fallback — O(1) lookup, avoids the If-Elif anti-pattern
  ✅ Clean 'exit' / 'quit' / 'bye' break command — the "kill command"
  ✅ 5+ intent categories in knowledge base
  ✅ Modular functions — each phase is its own responsibility
"""

import time
import random


# ─────────────────────────────────────────────────────────────────────────────
#  KNOWLEDGE BASE  (The White Box — fully traceable, zero hallucination risk)
#  Each key = a normalized user intent keyword
#  Each value = a list of responses (adds personality variety)
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE: dict[str, list[str]] = {

    # ── Greetings ──────────────────────────────────────────────────────────
    "hello": [
        "Hello! Welcome to DecodeLabs. How can I assist you today?",
        "Hi there! Ready to build something intelligent together?",
        "Hey! Great to see you. What's on your mind?",
    ],
    "hi": [
        "Hi! I'm ARIA, your Rule-Based AI assistant at DecodeLabs.",
        "Hello! How can I help you today?",
    ],
    "hey": [
        "Hey! What can I do for you?",
        "Hey there! ARIA at your service.",
    ],
    "good morning": [
        "Good morning! Hope you're ready to code something amazing today.",
        "Morning! The best time to build AI systems. What are we working on?",
    ],
    "good afternoon": [
        "Good afternoon! Halfway through the day — let's make it count.",
    ],
    "good evening": [
        "Good evening! Late session? Let's get it done.",
    ],

    # ── Identity & About ───────────────────────────────────────────────────
    "who are you": [
        "I'm ARIA — Automated Rule-based Intelligence Assistant, built by a "
        "DecodeLabs intern as Project 1. I operate on pure logic: no neural "
        "networks, no probabilities. Just rules.",
    ],
    "what are you": [
        "I'm a Rule-Based AI Chatbot — a 'white box' system where every "
        "decision is fully traceable. Input → Logic → Output. No mystery.",
    ],
    "your name": [
        "My name is ARIA — Automated Rule-based Intelligence Assistant.",
    ],
    "what is your name": [
        "I'm ARIA! Pleased to meet you.",
    ],

    # ── How are you ────────────────────────────────────────────────────────
    "how are you": [
        "I'm running at 100% deterministic efficiency — no bugs today!",
        "All logic gates are green. I'm doing great, thanks for asking!",
        "Operational and ready. How about you?",
    ],
    "how are you doing": [
        "Perfectly logical and fully operational. How can I help?",
    ],
    "are you okay": [
        "Zero errors, full uptime — I'm doing great!",
    ],

    # ── AI & Technology ────────────────────────────────────────────────────
    "what is ai": [
        "Artificial Intelligence is the simulation of human intelligence in "
        "machines. There are two main paradigms:\n"
        "  • System 1 (The Artist): Probabilistic — neural networks, LLMs\n"
        "  • System 2 (The Engineer): Deterministic — rule-based systems like me\n"
        "Before you master the chaos of a probability engine, you must master "
        "the precision of a logic engine.",
    ],
    "what is machine learning": [
        "Machine Learning is a subset of AI where systems learn patterns from "
        "data — rather than following explicit rules. It's System 1 (probabilistic). "
        "I'm System 2 (deterministic) — my rules are hard-coded and traceable.",
    ],
    "what is a chatbot": [
        "A chatbot is a program that simulates conversation. Rule-based chatbots "
        "like me use if-else logic and dictionary lookups. Advanced chatbots use "
        "LLMs (Large Language Models) — but even those need rule-based guardrails!",
    ],
    "what is a rule based system": [
        "A rule-based system is a 'white box' AI:\n"
        "  ✅ TRACEABILITY: Input → Logic → Output. No mystery.\n"
        "  ✅ SAFETY: Zero hallucination risk. 100% hard-coded.\n"
        "  ✅ COMPLIANCE: Essential for Finance & Healthcare industries.",
    ],
    "what is nlp": [
        "NLP (Natural Language Processing) is the field of AI that deals with "
        "understanding human language. I use the simplest form — exact keyword "
        "matching. Advanced NLP uses semantic embeddings and vector similarity.",
    ],
    "what is a llm": [
        "A Large Language Model (LLM) is a probabilistic AI trained on massive "
        "text datasets. They're powerful but can hallucinate. That's why real "
        "production systems layer rule-based guardrails (like me) on top of LLMs "
        "— think NVIDIA NeMo Guardrails or Llama Guard.",
    ],
    "what is python": [
        "Python is the dominant language in AI/ML development. Its readability, "
        "rich ecosystem (NumPy, Pandas, TensorFlow, PyTorch), and rapid "
        "prototyping make it ideal. This very chatbot is built in Python!",
    ],

    # ── DecodeLabs & Internship ────────────────────────────────────────────
    "what is decodelabs": [
        "DecodeLabs is a tech platform offering hands-on AI internship programs. "
        "Their philosophy: master foundational skills before advanced ones. "
        "Project 1 (me!) is the mandatory starting milestone for Batch 2026.",
    ],
    "tell me about this project": [
        "Project 1: Rule-Based AI Chatbot\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Goal    : Build a chatbot using pure control flow & logic\n"
        "Skills  : Control flow, decision-making logic, basic AI concepts\n"
        "Pattern : IPO Model → Sanitize → Dict Lookup (O(1)) → Respond\n"
        "Loop    : Infinite while True — exits only on 'exit'/'quit'/'bye'\n"
        "Why     : Before building systems that learn, master systems that rule.",
    ],
    "what is the ipo model": [
        "The IPO Model is the foundational blueprint for this chatbot:\n"
        "  INPUT  → Sanitization & Normalization (.lower().strip())\n"
        "  PROCESS → Intent Matching & State (O(1) Dictionary lookup)\n"
        "  OUTPUT → Response Generation (matched reply or fallback)\n"
        "It ensures every AI system is transparent and controlled.",
    ],

    # ── Help & Features ────────────────────────────────────────────────────
    "help": [
        "I can chat about these topics:\n"
        "  🤖 AI & Technology  — 'what is AI', 'what is ML', 'what is NLP'\n"
        "  🏢 DecodeLabs       — 'what is DecodeLabs', 'tell me about this project'\n"
        "  💬 Conversation     — 'how are you', 'who are you', 'tell me a joke'\n"
        "  ⏰ Time & Date      — 'what time is it', 'what is today'\n"
        "  🧮 Math             — 'calculate 5 + 3' (basic arithmetic)\n"
        "  🚪 Exit             — type 'exit', 'quit', or 'bye' to leave\n"
        "Ask me anything!",
    ],
    "what can you do": [
        "I'm a rule-based system, so my knowledge is explicitly defined. "
        "Type 'help' to see all topics I can discuss!",
    ],
    "commands": [
        "Available commands:\n"
        "  help       → Show all topics\n"
        "  history    → Show this conversation's history\n"
        "  clear      → Clear the screen\n"
        "  calculate  → Do basic math (e.g., 'calculate 10 + 5')\n"
        "  exit/quit/bye → End the session",
    ],

    # ── Fun & Personality ──────────────────────────────────────────────────
    "tell me a joke": [
        "Why do programmers prefer dark mode?\n"
        "Because light attracts bugs! 🐛",
        "Why did the AI go to therapy?\n"
        "It had too many deep issues! 🧠",
        "A SQL query walks into a bar, walks up to two tables and asks...\n"
        "'Can I JOIN you?' 😄",
        "Why don't scientists trust atoms?\n"
        "Because they make up everything — unlike rule-based systems!",
    ],
    "tell me a fun fact": [
        "The first chatbot, ELIZA, was built in 1966 at MIT — and it was "
        "rule-based, just like me!",
        "Python was named after Monty Python's Flying Circus, not the snake.",
        "The word 'bug' in programming comes from a real moth found in a "
        "Harvard computer in 1947 by Grace Hopper's team.",
        "Dictionary lookups in Python are O(1) — meaning they take the same "
        "time whether the dictionary has 10 items or 10 million!",
    ],
    "motivate me": [
        "Every expert was once a beginner. The first rule you define today "
        "is the foundation of your AI career. Keep building! 🚀",
        "An LLM without rules is a hallucination engine. YOU are building "
        "the skeleton that holds the intelligence. That matters.",
        "The best AI engineers understand deterministic systems before "
        "probabilistic ones. You're on the right path.",
    ],

    # ── Time & Date ────────────────────────────────────────────────────────
    "what time is it": ["__TIME__"],
    "current time": ["__TIME__"],
    "what is today": ["__DATE__"],
    "what day is it": ["__DATE__"],
    "what is the date": ["__DATE__"],

    # ── Gratitude ──────────────────────────────────────────────────────────
    "thank you": [
        "You're welcome! Keep building great things. 🛠️",
        "Anytime! That's what I'm here for.",
    ],
    "thanks": [
        "No problem! Is there anything else I can help with?",
        "Happy to help! 😊",
    ],
    "awesome": [
        "Glad you think so! What else can I do for you?",
    ],
    "great": [
        "Great! Let's keep the momentum going.",
    ],

    # ── Farewells (secondary — primary exit handled in loop) ───────────────
    "see you": [
        "See you! Keep coding and stay curious. 👋",
    ],
    "take care": [
        "You too! Remember: every rule you write today makes the AI safer tomorrow.",
    ],
    "goodbye": [
        "Goodbye! Your DecodeLabs journey is just beginning. 🚀",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
#  EXIT COMMANDS — The "Kill Command" (triggers clean break from while loop)
# ─────────────────────────────────────────────────────────────────────────────
EXIT_COMMANDS = {"exit", "quit", "bye", "quit()", "exit()"}

# ─────────────────────────────────────────────────────────────────────────────
#  FALLBACK — Default response when no intent is matched
# ─────────────────────────────────────────────────────────────────────────────
FALLBACK_RESPONSES = [
    "I don't understand that yet — I'm a rule-based system with a defined knowledge base. "
    "Try typing 'help' to see what I can discuss.",
    "Hmm, that's outside my current rule set. Type 'help' to explore what I know!",
    "No matching rule found for that input. As a white-box system, I only respond "
    "to explicitly defined intents. Type 'help' to see them.",
]


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1: INPUT & SANITIZATION
#  "HeLLo " → "hello"  |  "  WHAT IS AI  " → "what is ai"
# ─────────────────────────────────────────────────────────────────────────────

def sanitize(raw_input: str) -> str:
    """
    Normalize raw user input.
    Steps:
      1. .lower()  — case normalization (HELLO == hello == HeLLo)
      2. .strip()  — remove leading/trailing whitespace
    Returns a clean, lowercase string ready for dictionary lookup.
    """
    return raw_input.lower().strip()


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2: PROCESS — Intent Matching (O(1) Dictionary Lookup)
#  Uses dict.get() — the professional approach (not an if-elif ladder)
# ─────────────────────────────────────────────────────────────────────────────

def match_intent(clean_input: str) -> str:
    """
    Match sanitized input to a response using O(1) hash map lookup.

    Strategy:
      1. Exact match first  → O(1) dictionary.get()
      2. Keyword scan       → check if any key appears IN the user's input
         (handles "can you tell me a joke?" → matches "tell me a joke")
      3. Fallback           → graceful default response

    Returns a string response.
    """

    # ── Special computed responses ─────────────────────────────────────────
    # Math calculation: "calculate 5 + 3"
    if clean_input.startswith("calculate") or clean_input.startswith("calc"):
        return _handle_calculation(clean_input)

    # History command
    if clean_input == "history":
        return "__HISTORY__"

    # Clear command
    if clean_input == "clear":
        return "__CLEAR__"

    # ── Step 1: Exact match (O(1)) ─────────────────────────────────────────
    responses = KNOWLEDGE_BASE.get(clean_input)
    if responses:
        return _resolve_response(random.choice(responses))

    # ── Step 2: Keyword-in-input scan ─────────────────────────────────────
    # Allows partial matching: "hey can you help me" → matches "help"
    for key, responses in KNOWLEDGE_BASE.items():
        if key in clean_input:
            return _resolve_response(random.choice(responses))

    # ── Step 3: Fallback ───────────────────────────────────────────────────
    return random.choice(FALLBACK_RESPONSES)


def _resolve_response(response: str) -> str:
    """Resolve special placeholder tokens in responses."""
    if response == "__TIME__":
        return f"The current time is: {time.strftime('%H:%M:%S')} ⏰"
    if response == "__DATE__":
        return f"Today is: {time.strftime('%A, %B %d, %Y')} 📅"
    return response


def _handle_calculation(clean_input: str) -> str:
    """
    Handle basic arithmetic expressions.
    Supports: +, -, *, /
    Example: "calculate 10 + 5" → "Result: 15.0"
    """
    # Extract expression after 'calculate' or 'calc'
    expression = clean_input.replace("calculate", "").replace("calc", "").strip()
    if not expression:
        return "Please provide an expression. Example: 'calculate 10 + 5'"

    # Only allow safe characters: digits, operators, spaces, parentheses, dot
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "I can only handle basic arithmetic: +, -, *, /. Example: 'calculate 10 * 3'"

    try:
        result = eval(expression)  # safe here — input is strictly filtered
        return f"🧮 {expression} = {result}"
    except ZeroDivisionError:
        return "⚠️ Division by zero is undefined. Please try a valid expression."
    except Exception:
        return f"⚠️ Could not evaluate '{expression}'. Example format: 'calculate 10 + 5'"


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 3: OUTPUT — Response Generation & Feedback Loop
# ─────────────────────────────────────────────────────────────────────────────

def display_response(response: str) -> None:
    """Print the bot's response with the ARIA prefix."""
    print(f"\n  🤖 ARIA: {response}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  THE HEARTBEAT: THE INFINITE LOOP
#  "The organism stays alive until the Kill Command." — DecodeLabs Module 01
# ─────────────────────────────────────────────────────────────────────────────

def run_chatbot() -> None:
    """
    Main chatbot loop implementing the IPO model.

    while True:
        user_input  = get_input()          # Phase 1: Input
        if exit:    break                  # Kill Command
        clean       = sanitize(user_input) # Phase 1: Sanitization
        response    = match_intent(clean)  # Phase 2: Process
        display(response)                  # Phase 3: Output
    """

    # Conversation history — tracked in memory (session-scoped)
    history: list[dict] = []

    _print_banner()
    display_response(
        "Hello! I'm ARIA — your Rule-Based AI Assistant at DecodeLabs. 🤖\n"
        "  Type 'help' to see what I can do, or just start chatting!\n"
        "  Type 'exit', 'quit', or 'bye' to end the session."
    )

    # ── THE INFINITE LOOP — the heartbeat of the system ───────────────────
    while True:
        try:
            # ── PHASE 1: INPUT ─────────────────────────────────────────────
            raw_input = input("  👤 You: ").strip()

            # Guard: ignore empty input
            if not raw_input:
                print("  (Please type something — I'm listening!)\n")
                continue

            # ── KILL COMMAND — Clean exit ──────────────────────────────────
            if sanitize(raw_input) in EXIT_COMMANDS:
                print()
                display_response(
                    "Goodbye! 👋\n"
                    "  You've completed Project 1 of your DecodeLabs AI Internship.\n"
                    "  Remember: every rule you defined today is the foundation\n"
                    "  of the intelligence you'll build tomorrow. Keep going! 🚀"
                )
                break

            # ── PHASE 1: SANITIZATION ──────────────────────────────────────
            clean_input = sanitize(raw_input)

            # ── PHASE 2: PROCESS ───────────────────────────────────────────
            if clean_input == "__history__" or clean_input == "history":
                _show_history(history)
                continue

            if clean_input == "clear":
                _clear_screen()
                continue

            response = match_intent(clean_input)

            # ── PHASE 3: OUTPUT ────────────────────────────────────────────
            display_response(response)

            # Log to session history
            history.append({
                "user": raw_input,
                "aria": response,
                "time": time.strftime('%H:%M:%S')
            })

        except KeyboardInterrupt:
            print()
            display_response("Session interrupted. Goodbye! 👋")
            break


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    """Print the startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     ██████╗  █████╗  ██╗   ██╗ ██████╗  ██████╗                 ║
║     ██╔══██╗██╔══██╗ ██║   ██║██╔═══██╗██╔════╝                 ║
║     ██║  ██║███████║ ██║   ██║██║   ██║╚█████╗                  ║
║     ██║  ██║██╔══██║ ██║   ██║██║   ██║ ╚═══██╗                 ║
║     ██████╔╝██║  ██║ ╚██████╔╝╚██████╔╝██████╔╝                 ║
║     ╚═════╝ ╚═╝  ╚═╝  ╚═════╝  ╚═════╝ ╚═════╝                  ║
║                                                                  ║
║       DecodeLabs AI Internship — Batch 2026                      ║
║       Project 1: Rule-Based AI Chatbot                           ║
║       Architecture: IPO Model | O(1) Dictionary Lookup           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def _show_history(history: list[dict]) -> None:
    """Display the conversation history for this session."""
    if not history:
        print("\n  📋 No conversation history yet.\n")
        return
    print("\n  📋 SESSION HISTORY:")
    print("  " + "─" * 60)
    for i, entry in enumerate(history, 1):
        print(f"  [{entry['time']}] Turn {i}")
        print(f"    👤 You : {entry['user']}")
        print(f"    🤖 ARIA: {entry['aria'][:80]}{'...' if len(entry['aria']) > 80 else ''}")
        print()


def _clear_screen() -> None:
    """Clear the terminal screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_chatbot()
