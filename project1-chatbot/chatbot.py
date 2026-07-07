"""
Project 1: Rule-Based AI Chatbot
DecodeLabs Industrial Training Kit

Goal: predefined responses via if-else / dictionary lookup logic,
running in a continuous loop until an exit command.
"""

# Knowledge base: dictionary = O(1) lookup (faster & cleaner than if-elif chains)
responses = {
    "hello": "Hi there! I'm DecodeBot. How can I help you today?",
    "hi": "Hello! Ask me about DecodeLabs, or type 'help'.",
    "how are you": "I'm just code, but I'm running perfectly! How are you?",
    "what is your name": "I'm DecodeBot, a rule-based AI assistant.",
    "help": "You can say: hello, how are you, what is your name, bye.",
    "bye": "Goodbye! Thanks for chatting.",
    "exit": "Session ended. Goodbye!",
}

EXIT_COMMANDS = {"bye", "exit", "quit"}


def get_response(user_input: str) -> str:
    """Clean input, then look up a response with a safe fallback."""
    clean_input = user_input.lower().strip()
    return responses.get(clean_input, "I do not understand. Type 'help' for options.")


def run_chatbot():
    print("=== DecodeBot (Project 1) ===")
    print("Type 'help' for options, or 'bye' to exit.\n")

    while True:  # continuous loop = the chatbot's "heartbeat"
        user_input = input("You: ")
        clean_input = user_input.lower().strip()

        reply = get_response(user_input)
        print(f"Bot: {reply}")

        if clean_input in EXIT_COMMANDS:
            break  # kill command


if __name__ == "__main__":
    run_chatbot()
