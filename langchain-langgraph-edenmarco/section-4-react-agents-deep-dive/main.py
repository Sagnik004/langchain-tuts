from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_core.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from langchain_classic.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)
from langchain_core.agents import AgentAction, AgentFinish
from typing import Union, List

# Load environment variables from .env file
load_dotenv()


# Define the tools


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""

    print(f"get_text_length enter with text={text}")

    # Removing any non alphabetic characters first before returning length
    updated_text = text.strip("'\n").strip('"')
    return len(updated_text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found!")

def main():
    tools = [get_text_length]

    # https://smith.langchain.com/hub/hwchase17/react
    template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:
"""

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools=tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        stop_sequences=["\nObservation", "Observation", "Observation:"],
        # model_kwargs={"stop": ["\nObservation", "Observation", "Observation:"]}, # Gemini need this
    )

    # Creating our agent. First it takes input, but instead of hardcoding we are passing a lambda so that when
    # the invoke method is called with a dictionary like {"input": "..."} we can extract that value and use it.
    # We then pass the input to prompt to generate a PromptTemplate object when then is passed to the LLM. And,
    # finally we parse the output with the ReActSingleInputOutputParser to extract the LLM response.
    agent = (
        {"input": lambda x: x["input"]} | prompt | llm | ReActSingleInputOutputParser()
    )

    # Invoke the agent, and extract what tool to use and its input.
    # Sample response: tool='get_text_length' tool_input="'DOG'" log="To find the text length of 'DOG', I need to use the get_text_length tool.\n\nAction: get_text_length\nAction Input: 'DOG'"
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {"input": "What is the text length of 'DOG' in characters?"}
    )

    # If LLM response have a tool, tool_input, log etc. basically of type AgentAction, then go ahead and 
    # call the tool after extracting out the tool name and input to be passed to it.
    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"Observation={observation}")

if __name__ == "__main__":
    main()
