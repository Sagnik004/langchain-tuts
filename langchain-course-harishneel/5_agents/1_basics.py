# Reference: https://www.youtube.com/watch?v=JQx0iQqhXxU&list=PLNIQLFWpQMRU1Ayjc-LX2k01Jj7uDG0rU&index=28, https://youtu.be/Do4sh79gc_s?list=PLNIQLFWpQMRU1Ayjc-LX2k01Jj7uDG0rU

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent
from langchain.tools import tool
import datetime

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
llm_model = ChatOpenAI(
  model="gpt-4o",
  max_tokens=2048,
  temperature=0.6,
  timeout=30
)

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
  """ Returns the current date and time in the specified format """

  current_time = datetime.datetime.now()
  formatted_time = current_time.strftime(format)
  return formatted_time

tools = [get_system_time]

# User query
query = "What is the current time? I need only the time values."

agent = create_agent(
  model=llm_model, 
  tools=tools,
  system_prompt="You are a helpful assistant. Be concise and accurate.",
)

result = agent.invoke({"input": query})
print(result)
