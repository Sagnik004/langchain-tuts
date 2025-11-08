from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.tools.render import render_text_description
from langchain_openai import ChatOpenAI

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

    llm = ChatOpenAI(model="gpt-4o", temperature=0, stop_sequences=["\nObservation"])


if __name__ == "__main__":
    main()
