import os
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Initialize model
model = ChatOpenAI(
    model="stepfun/step-3.5-flash",
    temperature=0
)

tools = []
agent_executor = create_agent(model, tools)

# Chat function
def chat_with_ai(message, history):
    try:
        response = agent_executor.invoke(
            {"messages": [HumanMessage(content=message)]}
        )
        answer = response["messages"][-1].content
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# Create UI
demo = gr.ChatInterface(
    fn=chat_with_ai,
    title="🤖 AI Assistant",
    description="Powered by Step 3.5 Flash via OpenRouter",
)

if __name__ == "__main__":
    demo.launch()
