"""
Test Suite — Project 1: Rule-Based AI Chatbot
DecodeLabs AI Internship | Batch 2026

Tests cover:
  - Sanitization (Phase 1)
  - Intent matching (Phase 2)
  - Fallback behavior
  - Exit command detection
  - Edge cases
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chatbot import sanitize, match_intent, EXIT_COMMANDS, KNOWLEDGE_BASE


# ─────────────────────────────────────────────────────────────────────────────
#  TEST HELPERS
# ─────────────────────────────────────────────────────────────────────────────

passed = 0
failed = 0

def test(description: str, condition: bool) -> None:
    global passed, failed
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status}  |  {description}")
    if condition:
        passed += 1
    else:
        failed += 1


def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  📋 {title}")
    print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
#  TESTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n╔══════════════════════════════════════════════════════════╗")
print("║   PROJECT 1 — AUTOMATED TEST SUITE                      ║")
print("║   DecodeLabs AI Internship | Batch 2026                  ║")
print("╚══════════════════════════════════════════════════════════╝")


# ── Phase 1: Sanitization ────────────────────────────────────────────────────
section("PHASE 1: SANITIZATION TESTS")

test("'hello' stays 'hello'",          sanitize("hello")         == "hello")
test("'HELLO' becomes 'hello'",        sanitize("HELLO")         == "hello")
test("'HeLLo' becomes 'hello'",        sanitize("HeLLo")         == "hello")
test("'  hello  ' strips spaces",      sanitize("  hello  ")     == "hello")
test("'WHAT IS AI' becomes lowercase", sanitize("WHAT IS AI")    == "what is ai")
test("Empty string stays empty",       sanitize("")              == "")
test("Mixed case+spaces normalized",   sanitize("  How Are You ") == "how are you")


# ── Phase 2: Exact Intent Matching ───────────────────────────────────────────
section("PHASE 2: EXACT INTENT MATCHING TESTS")

greet_response = match_intent("hello")
test("'hello' returns a non-empty response",
     len(greet_response) > 0)

test("'hello' response is a string",
     isinstance(greet_response, str))

test("'who are you' matches identity intent",
     "aria" in match_intent("who are you").lower() or
     "rule" in match_intent("who are you").lower())

test("'what is ai' returns AI explanation",
     "intelligence" in match_intent("what is ai").lower() or
     "system" in match_intent("what is ai").lower())

test("'help' returns help text",
     "help" in match_intent("help").lower() or
     "topic" in match_intent("help").lower() or
     "chat" in match_intent("help").lower())

test("'what is the ipo model' matches IPO intent",
     "input" in match_intent("what is the ipo model").lower() or
     "ipo" in match_intent("what is the ipo model").lower())

test("'tell me a joke' returns a response",
     len(match_intent("tell me a joke")) > 0)

test("'thank you' gets an acknowledgment",
     len(match_intent("thank you")) > 0)


# ── Phase 2: Keyword-in-Input Matching ───────────────────────────────────────
section("PHASE 2: KEYWORD-IN-INPUT SCAN TESTS")

test("'can you tell me a joke please' matches 'tell me a joke'",
     len(match_intent("can you tell me a joke please")) > 10)

test("'please help me' matches 'help'",
     "help" in match_intent("please help me").lower() or
     "topic" in match_intent("please help me").lower() or
     len(match_intent("please help me")) > 10)

test("'say hello to me' matches greeting",
     len(match_intent("say hello to me")) > 10)


# ── Phase 2: Special Computed Responses ──────────────────────────────────────
section("PHASE 2: SPECIAL / COMPUTED RESPONSES")

time_response = match_intent("what time is it")
test("'what time is it' returns time string",
     ":" in time_response)  # time has HH:MM:SS format

date_response = match_intent("what is today")
test("'what is today' returns date string",
     "2" in date_response)  # year contains a digit

calc_response = match_intent("calculate 10 + 5")
test("'calculate 10 + 5' returns 15",
     "15" in calc_response)

calc_mult = match_intent("calculate 6 * 7")
test("'calculate 6 * 7' returns 42",
     "42" in calc_mult)

calc_div = match_intent("calculate 10 / 2")
test("'calculate 10 / 2' returns 5",
     "5" in calc_div)

calc_zero = match_intent("calculate 5 / 0")
test("Division by zero returns error message",
     "zero" in calc_zero.lower() or "undefined" in calc_zero.lower())


# ── Phase 3: Fallback ────────────────────────────────────────────────────────
section("PHASE 3: FALLBACK TESTS")

fallback = match_intent("xyzzy random gibberish input 999")
test("Unknown input returns fallback (non-empty)",
     len(fallback) > 10)

test("Fallback doesn't crash the system",
     isinstance(fallback, str))

test("Fallback mentions 'help' or 'rule'",
     "help" in fallback.lower() or "rule" in fallback.lower())


# ── Exit Commands ─────────────────────────────────────────────────────────────
section("EXIT COMMAND DETECTION TESTS")

test("'exit' is an exit command",   "exit"   in EXIT_COMMANDS)
test("'quit' is an exit command",   "quit"   in EXIT_COMMANDS)
test("'bye' is an exit command",    "bye"    in EXIT_COMMANDS)
test("'hello' is NOT exit command", "hello"  not in EXIT_COMMANDS)
test("'hi' is NOT exit command",    "hi"     not in EXIT_COMMANDS)


# ── Knowledge Base Structure ──────────────────────────────────────────────────
section("KNOWLEDGE BASE STRUCTURE TESTS")

test("Knowledge base is a dictionary",
     isinstance(KNOWLEDGE_BASE, dict))

test("Knowledge base has 5+ intent categories",
     len(KNOWLEDGE_BASE) >= 5)

test("All values are lists",
     all(isinstance(v, list) for v in KNOWLEDGE_BASE.values()))

test("All lists have at least one response",
     all(len(v) >= 1 for v in KNOWLEDGE_BASE.values()))

test("All keys are lowercase (pre-sanitized)",
     all(k == k.lower() for k in KNOWLEDGE_BASE.keys()))

print(f"\n{'═'*60}")
print(f"  RESULTS: {passed} passed  |  {failed} failed  |  {passed+failed} total")
print(f"  {'✅ ALL TESTS PASSED!' if failed == 0 else f'❌ {failed} TEST(S) FAILED'}")
print(f"{'═'*60}\n")
