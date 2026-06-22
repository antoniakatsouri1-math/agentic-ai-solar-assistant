"""
Router test harness (for the README's "Router test cases" table).

Runs at least 10 representative messages through the real classify_intent()
function and prints a markdown table comparing expected vs. actual route,
plus an accuracy summary. Requires a real ANTHROPIC_API_KEY in .env since
this makes real LLM calls.

Run: python scripts/test_router.py
"""
import sys

sys.path.insert(0, ".")

from src.router import classify_intent  # noqa: E402

TEST_CASES = [
    ("What is the weather in Athens tomorrow?", "weather"),
    ("Θα έχει ηλιοφάνεια στην Κρήτη αυτή την εβδομάδα;", "weather"),
    ("What's the cloud cover forecast for Thessaloniki?", "weather"),
    ("What are current electricity prices in Greece?", "search"),
    ("Υπάρχουν νέα προγράμματα επιδότησης φωτοβολταϊκών αυτή την περίοδο;", "search"),
    ("Any recent news on PV equipment prices?", "search"),
    ("What is net-metering and how does it work?", "rag"),
    ("Ποια είναι τα όρια ισχύος για αυτοπαραγωγούς;", "rag"),
    ("How do I get a Φωτοβολταϊκά στη Στέγη subsidy?", "rag"),
    ("What's the VAT rate on solar panels in Greece?", "rag"),
    ("What were total sales last month?", "sql"),
    ("Ποια κατηγορία προϊόντων είχε τα μεγαλύτερα έσοδα;", "sql"),
    ("How many 10kWh batteries have we sold?", "sql"),
    ("Explain what LangGraph is.", "general"),
]


def main():
    print(f"Running {len(TEST_CASES)} router test cases against the live LLM...\n")
    rows = []
    correct = 0
    for message, expected in TEST_CASES:
        actual = classify_intent(message)
        is_correct = actual == expected
        correct += is_correct
        rows.append((message, expected, actual, "✅" if is_correct else "❌"))

    print("| Input Message | Expected Route | Actual Route | Match |")
    print("|---|---|---|---|")
    for message, expected, actual, mark in rows:
        print(f"| {message} | {expected} | {actual} | {mark} |")

    print(f"\nAccuracy: {correct}/{len(TEST_CASES)} ({100 * correct / len(TEST_CASES):.0f}%)")


if __name__ == "__main__":
    main()
