from typing import List

from dotenv import load_dotenv
from langchain.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage

from callbacks import AgentCallbackHandler

# Load environment variables from .env file
load_dotenv()

# Define the tools


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip(
        '"'
    )  # stripping away non alphabetic characters just in case
    return len(text)


def find_tool_by_name(tools: List[BaseTool], tool_name: str) -> BaseTool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    tools = [get_text_length]
    llm = ChatOpenAI(model="gpt-4o", temperature=0, callbacks=[AgentCallbackHandler()])
    llm_with_tools = llm.bind_tools(tools)

    # Start conversation
    messages = [HumanMessage(content="What are the length of the words DOG, ZEBRA and HIPPOPOTAMUS?")]

    while True:
        ai_message = llm_with_tools.invoke(messages)

        # If the model decides to call tools, execute them and return results
        tool_calls = getattr(ai_message, "tool_calls", None) or []
        if len(tool_calls) > 0:
            messages.append(ai_message)
            for tool_call in tool_calls:
                # tool_call is typically a dict with keys: id, type, name, args
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("args", {})
                tool_call_id = tool_call.get("id")

                tool_to_use = find_tool_by_name(tools=tools, tool_name=tool_name)
                observation = tool_to_use.invoke(tool_args)
                print(f"observation={observation}")

                messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
            # Continue loop to allow the model to use the observations
            continue
        
        # No tool calls -> final answer
        print(ai_message.content)
        break
