from dotenv import load_dotenv
from langchain.tools import tool
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.tool_calling_agent.base import \
    create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# Load environment variables from .env file
load_dotenv()

# Define the tools


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y


def main():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    tools = [TavilySearch(), multiply]

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    res = agent_executor.invoke(
        {
            "input": "What is the weather in Dubai right now? Compare it with San Fransisco, and output should in in celsius."
        }
    )
    print(res)


if __name__ == "__main__":
    main()
