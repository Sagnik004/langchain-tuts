from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# Load environment variables from .env file
load_dotenv()

# Define the tools
tools = [TavilySearch()]

# Define the llm
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Pull the react prompt (special prompt template) from the hub
react_prompt = hub.pull("hwchase17/react")

# Create an agent that uses ReAct prompting (reasoning engine)
agent = create_react_agent(
    llm=llm, 
    tools=tools,
    prompt=react_prompt
)

# Agent runtime
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
chain = agent_executor

def main():
    result = chain.invoke(
        input={
            "input": "Search for 3 job postings for an AI engineer using LangChain in Bangalore, India on LinkedIn and list their details",
        }
    )
    print(result)


if __name__ == "__main__":
    main()
