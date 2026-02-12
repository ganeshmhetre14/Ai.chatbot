import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

def main():
    # Initialize model (Step 3.5 Flash via OpenRouter)
    model = ChatOpenAI(
        model="stepfun/step-3.5-flash",
        temperature=0
    )

    # No tools for now (simple chat agent)
    tools = []

    # Create agent
    agent_executor = create_agent(model, tools)

    print("Welcome! I'm your assistant. Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "quit":
            print("Goodbye 👋")
            break

        try:
            # Invoke agent (non-streaming to avoid blank issue)
            response = agent_executor.invoke(
                {"messages": [HumanMessage(content=user_input)]}
            )

            # Print last assistant message
            print("\nAssistant:", response["messages"][-1].content)

        except Exception as e:
            print("\nError:", str(e))


if __name__ == "__main__":
    main()
