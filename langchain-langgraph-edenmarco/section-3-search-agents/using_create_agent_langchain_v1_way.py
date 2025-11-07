from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage

from schemas import AgentResponse

# Load environment variables from .env file
load_dotenv()

# Initialize tools
tools = [TavilySearch()]

# Define the model
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Create a ReAct agent using the new create_agent interface
agent = create_agent(
    model,
    tools=tools,
    system_prompt="""Answer questions as best as you can. You have access to tools for searching the web.

Use the following apporoach:
1. Think about what information you need
2. Use available tools to gather that information
3. Reason about the results
4. Provide a comprehensive answer with sources

Be thorough and cite your sources.""",
    response_format=AgentResponse,
)


def main():
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Search for 3 active job postings for a Technology Architect or Solutions Architect in Bangalore India on LinkedIn and list their details",
                }
            ]
        }
    )
    structured_response = result.get("structured_response")
    print(structured_response)


if __name__ == "__main__":
    main()
