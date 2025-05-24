from dotenv import load_dotenv
load_dotenv()

import os
from typing import TypedDict, Annotated

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AnyMessage

# Use the new HuggingFaceEndpoint class
from langchain_huggingface import HuggingFaceEndpoint

# Import your tools as before
from tools import tools as base_tools
from retriever import guest_info_tool

tools = base_tools + [guest_info_tool]

# Set up Hugging Face LLM using the updated class
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=512,
    temperature=0.7,
)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Simplified assistant without tool-binding
def assistant(state: AgentState):
    query = state["messages"][-1].content
    response = llm.invoke(query)
    return {
        "messages": [
            HumanMessage(content=query),
            HumanMessage(content=response)
        ],
    }

builder = StateGraph(AgentState)
builder.add_node("assistant", assistant)
builder.set_entry_point("assistant")
alfred = builder.compile()

if __name__ == "__main__":
    messages = [HumanMessage(content="Who is Facebook and what's their most popular model?")]
    response = alfred.invoke({"messages": messages})
    print("🎩 Alfred's Response:")
    print(response["messages"][-1].content)
