import argparse

from src import memory
from src.graph import build_graph
def main():
    parser = argparse.ArgumentParser(description="Multi-tool agentic AI assistant")
    parser.add_argument(
        "--conversation-id",
        dest="conversation_id",
        default=None,
        help="Resume an existing conversation by id (omit to start a new one)",
    )
    args = parser.parse_args()

    memory.init_db()
    app = build_graph()

    if args.conversation_id:
        if memory.conversation_exists(args.conversation_id):
            conversation_id = args.conversation_id
            print(f"Resuming conversation {conversation_id}")
            history_so_far = memory.get_full_history(conversation_id)
            if history_so_far:
                print(f"({len(history_so_far)} prior messages loaded)\n")
        else:
            print(f"No conversation found with id {args.conversation_id} - starting a new one with that id.")
            conversation_id = memory.create_conversation(args.conversation_id)
    else:
        conversation_id = memory.create_conversation()
        print(f"Started new conversation: {conversation_id}")
        print("(pass --conversation-id to resume this session later)\n")

    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue


        history = memory.get_recent_turns(conversation_id, n_turns=memory.DEFAULT_MEMORY_WINDOW)
        memory.save_message(conversation_id, "user", user_input)

        result = app.invoke(
            {
                "user_input": user_input,
                "history": history,
                "conversation_id": conversation_id,
                "memory_window": memory.DEFAULT_MEMORY_WINDOW,
            }
        )
        response = result["response"]

        memory.save_message(conversation_id, "assistant", response)

        print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()
