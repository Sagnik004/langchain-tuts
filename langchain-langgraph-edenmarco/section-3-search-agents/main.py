from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

# Load environment variables from .env file
load_dotenv()

# Define the tools
tools = [TavilySearch()]

# Define the llm
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Pull the react prompt (special prompt template) from the hub
react_prompt = hub.pull("hwchase17/react")

# Output parser using Pydantic model
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

# Make use of our customized ReAct template with ability to send expected output format instructions
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
).partial(format_instructions=output_parser.get_format_instructions())

# Create an agent that uses ReAct prompting (reasoning engine)
agent = create_react_agent(
    llm=llm, tools=tools, prompt=react_prompt_with_format_instructions
)

# Agent runtime, extract the output coming from agent executor, parse the output into a Pydantic object
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)
extract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))

chain = agent_executor | extract_output | parse_output


def main():
    result = chain.invoke(
        input={
            "input": "Search for 3 job postings for an AI engineer using LangChain in Hyderabad India on LinkedIn and list their details",
        }
    )
    print(result)


if __name__ == "__main__":
    main()
